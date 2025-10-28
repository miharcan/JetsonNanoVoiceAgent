import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import subprocess
import json
import os

# --- CONFIG ---
DEVICE_INDEX = 11        # your USB mic index (check with python3 -m sounddevice)
DURATION = 5             # seconds to record
SAMPLERATE_IN = 48000
SAMPLERATE_OUT = 16000
RAW_FILE = "mic_test.wav"
PROC_FILE = "mic_16k.wav"

# Whisper.cpp config
WHISPER_MAIN = "/home/miharc/whisper.cpp/main"
WHISPER_MODEL = "/home/miharc/whisper.cpp/models/ggml-tiny.en.bin"

# llama.cpp-nano config (GPU-accelerated)
LLAMA_MAIN = "/home/miharc/llama.cpp-nano/main"
#LLAMA_MODEL = "/home/miharc/llama.cpp-nano/phi-2.Q4_0.gguf"
#LLAMA_MODEL = "/home/miharc/llama.cpp-nano/phi-2.Q3_K_M.gguf"
LLAMA_MODEL = "/home/miharc/llama.cpp-nano/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

# GPU and sampling settings
GPU_LAYERS = 2
N_TOKENS = 100
TEMP = 0.7
TOP_K = 40
TOP_P = 0.9


def record_audio():
    """Record from USB mic at 48kHz and resample to 16kHz"""
    print(f"🎤 Recording {DURATION}s from device {DEVICE_INDEX} at {SAMPLERATE_IN} Hz...")
    audio = sd.rec(
        int(DURATION * SAMPLERATE_IN),
        samplerate=SAMPLERATE_IN,
        channels=1,
        dtype="float32",
        device=DEVICE_INDEX
    )
    sd.wait()
    sf.write(RAW_FILE, audio, SAMPLERATE_IN, subtype="PCM_16")
    print(f"✅ Saved raw recording to {RAW_FILE}")

    # Resample 48 kHz → 16 kHz
    print("🔄 Resampling to 16kHz...")
    audio_resampled = resample_poly(audio.flatten(), up=1, down=3)
    sf.write(PROC_FILE, audio_resampled, SAMPLERATE_OUT, subtype="PCM_16")
    print(f"✅ Saved resampled audio to {PROC_FILE}")
    return PROC_FILE


def transcribe(filepath):
    """Run Whisper.cpp transcription"""
    print("📝 Transcribing with Whisper.cpp...")
    cmd = [WHISPER_MAIN, "-m", WHISPER_MODEL, "-f", filepath, "-nt"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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


def ask_llama(prompt):
    """Send prompt to llama.cpp-nano CLI (GPU)"""
    print("🦙 Running llama.cpp-nano with GPU acceleration...")

    cmd = [
        LLAMA_MAIN,
        "-m", LLAMA_MODEL,
        "--gpu-layers", str(GPU_LAYERS),
        "--threads", "3",
        "--ctx-size", "256",
        "-n", "200",
        "--temp", str(TEMP),
        "--top-k", str(TOP_K),
        "--top-p", str(TOP_P),
        #"-p", f"<|user|> {prompt}\n<|end|>\n<|assistant|>"
        "-p", f"[INST] {prompt} [/INST]"
    ]

    # use run() instead of Popen for now
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    reply = result.stdout.strip()
    #print(reply)
    return reply



if __name__ == "__main__":
    try:
        wavfile = record_audio()
        text = transcribe(wavfile)

        # Cleanup audio files after processing
        for f in [RAW_FILE, PROC_FILE]:
            if os.path.exists(f):
                os.remove(f)
                print(f"🧹 Deleted {f}")

        if text:
            reply = ask_llama(text)
            print("\nFull reply:\n", reply)
        else:
            print("⚠️ No usable speech, skipping Llama.")
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user.")

