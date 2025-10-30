# JetsonNanoVoiceAgent

![Jetson Nano](https://developer.nvidia.com/sites/default/files/akamai/embedded/images/jetson-nano-dev-kit-top-angle-800.jpg)

**Jetson Nano LLM** explores running lightweight Large Language Models (LLMs) and speech models directly on the NVIDIA Jetson Nano — a 4GB ARM-based edge AI board with a 128-core Maxwell GPU.

---

## ⚙️ Overview

This project tests **llama.cpp**, **whisper.cpp**, and other quantized LLMs for **offline conversational AI**, **speech-to-text**, and **edge reasoning**.  
Focus areas include **quantization**, **CUDA acceleration**, and **resource optimization** for real-world on-device AI.

| Component | Spec |
|------------|------|
| GPU | 128-core Maxwell |
| CPU | Quad-core ARM Cortex-A57 |
| Memory | 4 GB LPDDR4 |
| CUDA | 10.2 / 11.x |
| OS | Ubuntu 18.04 / 20.04 (JetPack 4.x) |

---

## ⚠️ Key Challenges

- **Memory limits** → models must be 4-bit or 8-bit quantized  
- **Low GPU compute** → use partial CUDA offload and short context sizes  
- **Compatibility issues** → prefer llama.cpp / whisper.cpp over heavy Python stacks  

---

## 🚀 Tips

- Use quantized models (`Q4_K_M`, `Q5_1`)  
- Enable CUDA flags:
  ```bash
  export GGML_CUDA_FORCE_MMQ=1
  export CUDA_USE_TENSOR_CORES=1
