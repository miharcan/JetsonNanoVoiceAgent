import sounddevice as sd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import subprocess
import requests
import json

# --- CONFIG ---
DEVICE_INDEX = 11    # your USB mic (AK5371)
DURATION = 5
SAMPLERATE_IN = 48000
SAMPLERATE_OUT = 16000
RAW_FILE = "mic_test.wav"
PROC_FILE = "mic_16k.wav"

WHISPER_MAIN = "../whisper.cpp/main"
WHISPER_MODEL = "../whisper.cpp/models/ggml-tiny.en.bin"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"   # change if your Ollama model has a different name


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

    # Resample 48k → 16k
    print("🔄 Resampling to 16kHz...")
    audio_resampled = resample_poly(audio.flatten(), up=1, down=3)
    sf.write(PROC_FILE, audio_resampled, SAMPLERATE_OUT, subtype="PCM_16")
    print(f"✅ Saved resampled audio to {PROC_FILE}")
    return PROC_FILE


def transcribe2(filepath):
    """Run Whisper.cpp transcription"""
    print("📝 Transcribing with Whisper.cpp...")
    cmd = [WHISPER_MAIN, "-m", WHISPER_MODEL, "-f", filepath, "-nt"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    text = ""
    for line in result.stdout.splitlines():
        if "]" in line and "-->" in line:
            part = line.split("]", 1)[1].strip()
            text += " " + part
    text = text.strip()
    print("🗣️ You said:", text if text else "⚠️ No speech recognized.")
    return text
    
def transcribe(filepath):
    """Run Whisper.cpp transcription"""
    print("📝 Transcribing with Whisper.cpp...")
    cmd = [WHISPER_MAIN, "-m", WHISPER_MODEL, "-f", filepath, "-nt"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Collect lines that aren't empty and aren't timing headers
    lines = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "-->" in line:  # has timestamp
            # take text part only
            if "]" in line:
                line = line.split("]", 1)[1].strip()
            lines.append(line)
        elif not line.startswith("whisper_") and not line.startswith("system_info"):
            # fallback: keep other text lines
            lines.append(line)

    text = " ".join(lines).strip()
    print("🗣️ You said:", text if text else "⚠️ No speech recognized.")
    return text



def ask_ollama(prompt):
    """Send text to Ollama and stream the reply"""
    print("🤖 Asking Ollama...")
    resp = requests.post(
        OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}, stream=True
    )
    output = ""
    for line in resp.iter_lines():
        if line:
            msg = json.loads(line.decode("utf-8"))
            if "response" in msg:
                print(msg["response"], end="", flush=True)
                output += msg["response"]
            if msg.get("done", False):
                break
    print("\n✅ Ollama reply complete")
    return output


if __name__ == "__main__":
    wavfile = record_audio()
    text = transcribe(wavfile)
    if text:
        reply = ask_ollama(text)
        print("\nFull reply:\n", reply)
    else:
        print("⚠️ No usable speech, skipping Ollama.")

