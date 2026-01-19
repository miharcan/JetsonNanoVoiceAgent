import argparse
import json
import subprocess
from pathlib import Path

import requests
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly


def _default_paths():
    """Reasonable defaults if you keep whisper.cpp as a sibling folder."""
    here = Path(__file__).resolve().parent
    whisper_main = (here / ".." / "whisper.cpp" / "main").resolve()
    whisper_model = (here / ".." / "whisper.cpp" / "models" / "ggml-tiny.en.bin").resolve()
    return {
        "whisper_main": str(whisper_main),
        "whisper_model": str(whisper_model),
    }


def parse_args():
    defaults = _default_paths()

    p = argparse.ArgumentParser(
        description="Record from a mic, transcribe via whisper.cpp, then query a local Ollama model.")

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

    p.add_argument("--ollama-url", default="http://localhost:11434/api/generate",
                   help="Ollama generate endpoint")
    p.add_argument("--ollama-model", default="gemma3:1b",
                   help="Ollama model name (must be pulled locally)")

    return p.parse_args()


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
    # common Jetson setup: 48k -> 16k (downsample by 3)
    audio_resampled = resample_poly(audio.flatten(), up=args.samplerate_out, down=args.samplerate_in)
    sf.write(args.proc_file, audio_resampled, args.samplerate_out, subtype="PCM_16")
    print(f"✅ Saved resampled audio to {args.proc_file}")

    return args.proc_file


def _require_file(path: str, label: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{label} not found at: {p}\n"
            "Tip: pass the correct path via CLI (see --help), or update your repo layout.")


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
            f"Stderr: {result.stderr.strip()}")

    # Parse Whisper output
    lines = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "-->" in line:  # timestamped line
            if "]" in line:
                line = line.split("]", 1)[1].strip()
            lines.append(line)
        elif not line.startswith("whisper_") and not line.startswith("system_info"):
            lines.append(line)

    text = " ".join(lines).strip()
    print("🗣️ You said:", text if text else "⚠️ No speech recognized.")
    return text


def ask_ollama(args, prompt: str) -> str:
    """Send text to Ollama and stream the reply."""
    print(f"🤖 Asking Ollama model: {args.ollama_model}")

    try:
        resp = requests.post(
            args.ollama_url,
            json={"model": args.ollama_model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=30,
        )
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not connect to Ollama at {args.ollama_url}.\n"
            "Tip: ensure Ollama is running and the URL is correct.\n"
            f"Details: {e}")

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {resp.status_code}.\n"
            f"Response: {resp.text[:500]}")

    output = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            msg = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            # Sometimes servers can emit keep-alives / partial lines
            continue

        if "response" in msg:
            print(msg["response"], end="", flush=True)
            output += msg["response"]
        if msg.get("done", False):
            break

    print("\n✅ Ollama reply complete")
    return output.strip()


def main():
    args = parse_args()

    wavfile = record_audio(args)
    text = transcribe(args, wavfile)

    if text:
        reply = ask_ollama(args, text)
        print("\nFull reply:\n", reply)
    else:
        print("⚠️ No usable speech, skipping Ollama.")


if __name__ == "__main__":
    main()
