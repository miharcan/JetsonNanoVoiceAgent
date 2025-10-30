# 🦙 Jetson Nano Voice Assistant (LLM + Whisper)

![Jetson Nano](https://developer.nvidia.com/sites/default/files/akamai/embedded/images/jetson-nano-dev-kit-top-angle-800.jpg)

A lightweight **voice-to-LLM pipeline** running fully offline on the **NVIDIA Jetson Nano**.  
It records audio from a USB mic, transcribes speech using **Whisper.cpp**, and replies with **Llama.cpp-nano** (GPU-accelerated).

---

## ⚙️ Overview

This project demonstrates end-to-end **speech-to-AI conversation** on an edge device.  
It’s built for experimentation with **TinyLlama**, **Phi-2**, and other small quantized LLMs optimized for low memory (4 GB).

### 🧩 Core Components
- 🎙️ **Audio Input** – `sounddevice` + `soundfile` for 48 kHz mic recording  
- 🔄 **Preprocessing** – resample to 16 kHz using `scipy.signal.resample_poly`  
- 🗣️ **Speech Recognition** – [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp) (Tiny English model)  
- 🦙 **LLM Inference** – [`llama.cpp`](https://github.com/ggerganov/llama.cpp) compiled for Jetson Nano GPU  
- 💬 **Pipeline** – Record → Transcribe → Generate LLM reply → Print to console  

---

## 🧱 Requirements

- Jetson Nano (4 GB) with JetPack 4.x  
- CUDA enabled (`export GGML_CUDA_FORCE_MMQ=1`)  
- Python 3.8+ with:
  ```bash
  pip install sounddevice soundfile numpy scipy
