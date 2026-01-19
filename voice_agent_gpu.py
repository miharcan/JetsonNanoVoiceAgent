import argparse
import os
import subprocess
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly


def _default_paths():
    """Reasonable defaults if you keep whisper.cpp and llama.cpp-nano as sibling folders."""
    here = Path(__file__).resolve().parent
    whisper_main = (here / ".." / "whisper.cpp" / "main").resolve()
    whisper_model = (here / ".." / "whisper.cpp" / "models" / "ggml-tiny.en.bin").resolve()

    llama_main = (here / ".." / "llama.cpp-nano" / "main").resolve()
    # You will almost certainly override this, but a sibling default helps.
    llama_model = (here / ".." / "llama.cpp-nano" / "models" / "tinyllama-1.1b-chat-v1.0.Q4_0.gguf").resolve()

    return {
        "whisper_main": str(whisper_main),
        "whisper_model": str(whisper_model),
        "llama_main": str(llama_main),
        "llama_model": str(llama_model),
    }


def parse_args():
    defaults = _default_paths()

    p = argparse.ArgumentParser(
        description="Record from a mic, transcribe via whisper.cpp, then run llama.cpp-nano (GPU)."
    )

    p.add_argument("--device-index", type=int, default=-1,
                   help="sounddevice input device index. Use -1 to use the default input device.")
    p.add_argument("--duration", type=float, default=5.0, help="seconds to record")

    p.add_argument("--samplerate-in", type=int, default=48000, help="mic sample rate")
    p.add_argument("--samplerate-out", type=int, default=16000, help="resampled output rate")

    p.add_argument("--raw-file", default="mic_test.wav", help="raw recording path")
    p.add_argument("--proc-file", default="mic_16k.wav", help="resampled wav path")

    p.add_argument("--whisper-main", default=defaults["whisper_main"],
                   help="path to whisper.cpp binary (e.g. ../whisper.cpp/main)")
    p.add_argument("--whisper-model", default=defaults["whisper_model"],
                   help="path to whisper.cpp model (e.g. ../whisper.cpp/models/ggml-tiny.en.bin)")

    p.add_argument("--llama-main", default=defaults["llama_main"],
                   help="path to llama.cpp-nano binary (e.g. ../llama.cpp-nano/main)")
    p.add_argument("--llama-model", default=defaults["llama_model"],
                   help="path to a .gguf model file")

    p.add_argument("--gpu-layers", type=int, default=2, help="number of layers to offload")
    p.add_argument("--threads", type=int, default=3, help="CPU threads")
    p.add_argument("--ctx-size", type=int, default=256, help="context window")
    p.add_argument("--n-tokens", type=int, default=200, help="max tokens to generate")

    p.add_argument("--temp", type=float, default=0.7, help="sampling temperature")
    p.add_argument("--top-k", type=int, default=40, help="top-k")
    p.add_argument("--top-p", type=float, default=0.9, help="top-p")

    p.add_argument("--keep-wavs", action="store_true", help="do not delete the recorded wavs")

    return p.parse_args()


def _require_file(path: str, label: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{label} not found at: {p}\n"
            "Tip: pass the correct path via CLI (see --help), or update your repo layout."
        )


def record_audio(args) -> str:
    """Record from USB mic at samplerate_in and resample to samplerate_out."""
    device = None if args.device_index is None or args.device_index < 0 else args.device_index

    print(f"🎤 Recording {args.duration}s from device {device if device is not None else 'DEFAULT'} at {args.samplerate_in} Hz...")
    audio = sd.rec(
        int(args.duration * args.samplerate_in),
        samplerate=args.samplerate_in,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()

    sf.write(args.raw_file, audio, args.samplerate_in, subtype="PCM_16")
    print(f"✅ Saved raw recording to {args.raw_file}")

    print(f"🔄 Resampling to {args.samplerate_out} Hz...")
    audio_resampled = resample_poly(audio.flatten(), up=args.samplerate_out, down=args.samplerate_in)
    sf.write(args.proc_file, audio_resampled, args.samplerate_out, subtype="PCM_16")
    print(f"✅ Saved resampled audio to {args.proc_file}")

    return args.proc_file


def transcribe(args, filepath: str) -> str:
    """Run Whisper.cpp transcription."""
    _require_file(args.whisper_main, "WHISPER_MAIN")
    _require_file(args.whisper_model, "WHISPER_MODEL")

    print("📝 Transcribing with Whisper.cpp...")
    cmd = [args.whisper_main, "-m", args.whisper_model, "-f", filepath, "-nt"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "Whisper.cpp failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr: {result.stderr.strip()}"
        )

    # Parse Whisper output
    lines = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "-->" in line:
            if "]" in line:
                line = line.split("]", 1)[1].strip()
            lines.append(line)
        elif not line.startswith("whisper_") and not line.startswith("system_info"):
            lines.append(line)

    text = " ".join(lines).strip()
    print("🗣️ You said:", text if text else "⚠️ No speech recognized.")
    return text


def _extract_reply(stdout: str) -> str:
    """Best-effort extraction of the assistant reply from llama.cpp-style CLI output."""
    s = stdout.strip()
    if not s:
        return s

    # If the prompt uses [INST] ... [/INST], the model reply is often after the closing token.
    marker = "[/INST]"
    if marker in s:
        s = s.split(marker, 1)[1].strip()

    # Remove common end tokens
    for tok in ("</s>", "<|endoftext|>"):
        s = s.replace(tok, "").strip()

    return s


def ask_llama(args, prompt: str) -> str:
    """Send prompt to llama.cpp-nano CLI (GPU)."""
    _require_file(args.llama_main, "LLAMA_MAIN")
    _require_file(args.llama_model, "LLAMA_MODEL")

    print("🦙 Running llama.cpp-nano with GPU acceleration...")

    cmd = [
        args.llama_main,
        "-m",
        args.llama_model,
        "--gpu-layers",
        str(args.gpu_layers),
        "--threads",
        str(args.threads),
        "--ctx-size",
        str(args.ctx_size),
        "-n",
        str(args.n_tokens),
        "--temp",
        str(args.temp),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "-p",
        f"[INST] {prompt} [/INST]",
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "llama.cpp-nano failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr: {result.stderr.strip()}"
        )

    return _extract_reply(result.stdout)


def main():
    args = parse_args()

    try:
        wavfile = record_audio(args)
        text = transcribe(args, wavfile)

        if not args.keep_wavs:
            for f in [args.raw_file, args.proc_file]:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"🧹 Deleted {f}")

        if text:
            reply = ask_llama(args, text)
            print("\nFull reply:\n", reply)
        else:
            print("⚠️ No usable speech, skipping Llama.")

    except KeyboardInterrupt:
        print("\n❌ Interrupted by user.")


if __name__ == "__main__":
    main()
