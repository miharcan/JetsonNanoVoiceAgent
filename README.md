# 🦙 Jetson Nano Voice Assistant (Whisper.cpp + LLM)

A lightweight **voice-to-LLM pipeline** intended for the **NVIDIA Jetson Nano**.
It records audio from a USB mic, transcribes speech with **whisper.cpp**, and then sends the text to an LLM backend.

This repo includes **two** runnable scripts:

- `voice_agent_gpu.py`: Whisper.cpp → **llama.cpp-nano** (CLI, GPU-offload)
- `voice_agent_cpu.py`: Whisper.cpp → **Ollama** (local HTTP server)

---

## What you need

- Jetson Nano (commonly JetPack 4.x) or any Linux box with a working audio input device
- Python 3.8+
- System audio deps for `sounddevice` (often PortAudio)
  - Ubuntu/Debian: `sudo apt-get install -y portaudio19-dev`

### Whisper.cpp
You need a built Whisper.cpp binary and a model file.

Typical layout (recommended):

```
parent-folder/
  JetsonNanoVoiceAgent/
  whisper.cpp/
    main
    models/ggml-tiny.en.bin
```

### LLM backend (choose one)

#### Option A: llama.cpp-nano (GPU script)
You need a built `llama.cpp-nano` binary and a `.gguf` model.

Example layout:

```
parent-folder/
  JetsonNanoVoiceAgent/
  llama.cpp-nano/
    main
    models/<your-model>.gguf
```

#### Option B: Ollama (CPU script)
Install/run Ollama locally and pull a model (example):

- Start Ollama (varies by install)
- Pull a model: `ollama pull gemma3:1b`

---

## Install

```bash
pip install -r requirements.txt
```

---

## Find your microphone device index

```bash
python3 -m sounddevice
```

Pick the input device index you want to use, then pass it with `--device-index`.
Use `--device-index -1` to use the system default input device.

---

## Run

### GPU path (Whisper.cpp + llama.cpp-nano)

```bash
python3 voice_agent_gpu.py \
  --device-index 11 \
  --duration 5 \
  --whisper-main ../whisper.cpp/main \
  --whisper-model ../whisper.cpp/models/ggml-tiny.en.bin \
  --llama-main ../llama.cpp-nano/main \
  --llama-model ../llama.cpp-nano/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --gpu-layers 2
```

### CPU path (Whisper.cpp + Ollama)

```bash
python3 voice_agent_cpu.py \
  --device-index 11 \
  --duration 5 \
  --whisper-main ../whisper.cpp/main \
  --whisper-model ../whisper.cpp/models/ggml-tiny.en.bin \
  --ollama-model gemma3:1b
```

If your Ollama server isn’t on the default URL, override it:

```bash
python3 voice_agent_cpu.py --ollama-url http://localhost:11434/api/generate
```

---

## Troubleshooting

- **`WHISPER_MAIN not found` / `WHISPER_MODEL not found`**
  - Pass the correct paths with `--whisper-main` and `--whisper-model`.
- **Ollama connection errors**
  - Ensure Ollama is running and you’re using the correct `--ollama-url`.
- **No audio / wrong mic**
  - Run `python3 -m sounddevice` and try a different `--device-index`.

---

## Notes

- Scripts record at 48 kHz by default and resample to 16 kHz for Whisper.
- The GPU script prints a best-effort “clean” reply by stripping common prompt markers from llama.cpp output.
