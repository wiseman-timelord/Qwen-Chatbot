# Qwen-Windows-Gguf
Status - Beta; needs more testing, but seems ok. 

## Description
A high-quality local chat interface for Qwen GGUF models on Windows 10 (WSL not required), using Python 3.12. An optimal number of features for a ChatBot, as well as, dynamic buttons/panels on the interface and websearch and RAG and TTS and archiving of sessions, and all on local models, so no imposed, limitations or guidelines (model dependent). This tool provides a comparable interface to premium non-agentic AI services, where the configuration is intended to be intelligent, without over-complication. The program uses offline libraries (apart from websearch) instead of, online services or repeat download or registration. This program is a fork of  [Chat-Gradio-Gguf](https://github.com/wiseman-timelord/Chat-Gradio-Gguf), which tends to be extremely hard for AI to fix in my last attempt, hence I have focused the program on my specific needs. Main thing, Qwen-Windows-Gguf does NOT have the extra-blank-lines bug that plagued development of Chat-Gradio-Gguf, so major victory/progression in my mind.

### Media
- As you can see the program works, however, the Qwen3.6-14B-FableVibes models are naff, this is probably because there is no such thing as Qwen 3.6/3.7 in 9B/14B, so stick to Qwen3.5 in 9b/14B, or use a Qwen 3.6/3.8 in 27b/35B, is my advice. I wouldnt bother with 4B models, as they are somewhat a joke for answers requiring accuracy. 
![Media_Missing](https://github.com/wiseman-timelord/Qwen-Windows-Gguf/blob/main/Media/Interactions_Page.jpg)

### Features
- **Qt-Web Custom Browser**: The interface uses Qt6 WebEngine with Gradio, appearing as a regular application; your default browser is untouched.
- **GPU Support**: Vulkan (binary download or compile), with GPU selection for multi CPU/GPU setups; CPU-only mode also supported.
- **Research-Grade Tools**: RAG, web search, chunking, THINK-phase streaming, Markdown formatting, and file attachments.
- **Text To Speech**: Kokoro TTS for realistic reading of output, filtered of symbols/tags/thinking appropriately.
- **Common File Support**: Handles `.bat`, `.py`, `.ps1`, `.txt`, `.json`, `.yaml`, `.psd1`, `.xaml`, and other common formats.
- **Configurable Context**: Set model context to 8192-138072, and batch output to 256-8192.
- **Enhanced Interface Controls**: Load/unload models, manage sessions, shutdown, and configure settings.
- **Highly Customizable UI**: 4-16 session history slots, 2-10 file slots, session log 450-1300px height, 2-8 lines of input.
- **Collapsable Left/Right Columns**: Like modern AI interfaces, with concise collapsed view for commonly used buttons.
- **Asynchronous Response Stream**: Separate thread with its own event loop, so response chunks are processed without blocking the Gradio UI.
- **Thinking/Reasoning Compatible**: Qwen3/3.5/3.6 `<think>` streaming handled natively; dynamic prompt system adapts for, uncensored, nsfw, chat, code.
- **Virtual Environment**: Isolated Python setup in `.venv` with `data` directory for constants, vectors, temp, and history.

## Requirements
- **Windows 10 22H2** — no Linux/dual-mode code (may work on Windows 11, but no version-specific code).
- **Python 3.12** — no version-specific code for other Python versions (3.13 unsupported: Kokoro TTS requires <3.13).
- **Qwen v3 to v3.8, 1B-35B, GGUF** — including variants such as HuiHui abliterated/uncensored builds.

## Usage
1. Right-click `Qwen-Windows-Gguf.bat` and Run as Administrator.
2. Select `2. Run Installation` to create the `.venv` and install the llama.cpp backend and dependencies.
3. Select `1. Run Main Program` to launch the interface.
4. On the Configuration page, set your model folder, pick the model, and load it.
5. Back on Interactions page, use the features and run your iterations.

## Structure
```
project_root/
│ Qwen-Windows-Gguf.bat      (menu: run / install)
│ installer.py               (standalone installer script)
│ launcher.py                (entry point: startup, shutdown)
├── media/
├── scripts/
│ └── __init__.py
│ └── display.py             (interface, Qt6 browser)
│ └── inference.py           (model loading & inference)
│ └── configure.py           (globals, constants, maps)
│ └── tools.py               (web search, TTS)
│ └── utility.py             (general functions, RAG)
├── data/                    (created by installer)
└── .venv/                   (created by installer)
```

## Credits
- By WiseMan-TimeLord; forked from [Chat-Gradio-Gguf](https://github.com/wiseman-timelord/Chat-Gradio-Gguf).
- [llama.cpp](https://github.com/ggml-org/llama.cpp), [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), [Gradio](https://gradio.app), [Kokoro TTS](https://github.com/hexgrad/kokoro).
