# Qwen-Chatbot
Status - Beta. 
- Had some good upgrades lately, there is possibility for more testing/improving/innovating. 

## Description
A high-quality local chat interface for Qwen GGUF models on Windows 10/11 (WSL not required), using Python 3.10-3.12. An optimal number of features for a ChatBot, as well as, dynamic buttons/panels on the interface and websearch and RAG and TTS and archiving of sessions, and all on local models, so no imposed, limitations or guidelines (model dependent). This tool provides a comparable interface to premium non-agentic AI services, where the configuration is intended to be intelligent, without over-complication. The program uses offline libraries (apart from websearch) instead of, online services or repeat download or registration. This program is a fork of  [Chat-Gradio-Gguf](https://github.com/wiseman-timelord/Chat-Gradio-Gguf), which tends to be extremely hard for AI to fix in my last attempt, hence I have focused the program towards Llama.cpp on Windows with GGUF, because this is optimal for Vulkan which runs on any GPU. Also to note Qwen-Windows-Gguf does NOT have the extra-blank-lines bug that plagued development of Chat-Gradio-Gguf, and because of development of Qwen-Windows-Gguf, I can now fix Chat-Gradio-Gguf.

### Media
- Here I am testing the Qwen3.8-9B-Abliterated-25-GGUF model, you can see the recently simplified interface layout (v0.15)... 
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
- **Windows 10-11** — Its for windows, and this time there is no Linux mode code.
- **Python 3.10-3.12** — no version-specific code for other Python versions (3.13 unsupported: Kokoro TTS requires <3.13).
- **Qwen v3 to v3.8, 1B-35B, GGUF** — See models section below.

### Models Support 
Remember while you can always update model support via editing the llama.cpp version in the installer.py script ie b8943 or whatever the [latest release](https://github.com/ggml-org/llama.cpp/releases) is, but there are also identifiers for model detection within the main program, unsure currently if future models will identify correctly, however...
- Specifically only Qwen versions 3/3.5/3.6/3.8 in 4B-35BA3B, including, Abliterated, Heretic, HuiHui, Uncensored etc. 
- I advise Gguf files in q5 quantization, unless this would cause excessive overflow into system ram, in which case use q4.
- To find a model, for example [search HuggingFace.Co](https://huggingface.co/models?search=qwen%203.6%20gguf), and locate a gguf file to fit your vram, for example [this one is interesting](https://huggingface.co/tvall43/Qwen3.6-14B-A3B-FableVibes-GGUF/tree/main), only 3B loaded at a time and smaller than 35B.
- Generally use a smaller Context Length ie 32768 for a websearch, larger context tend to be slower and use more memory, but there are options for up to 131072 tokens Context_Length.
- Gguf/Llama.Cpp does not have Sliding Context window, but the program does have RAG and some other thing, unsure. 
- The newer Qwen models have larger Batch_Output, while in program is max 8192 tokens, options can be edited in configure.py.

## Usage
1. Right-click `Qwen-Chatbot.bat` and Run as Administrator.
2. Select `2. Run Installation` to create the `.venv` and install the llama.cpp backend and dependencies.
3. Select `1. Run Main Program` to launch the interface.
4. On the Configuration page, set your model folder, pick the model, and load it.
5. Back on Interactions page, use the features and run your iterations.

## Structure
```
project_root/
│ Qwen-Chatbot.bat      (menu: run / install)
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

### Development
- Think/NoThink button turned out to be bad idea, because typically quantized finetuned models under 50B are designed to either be a Thinking or Non-Thinking model, and thinking models simply, do not work well without or are unable to stop using, thinking mode. What we need is a switch between 2 pre-configured models on configruation page, where there would be a row with 2 columns for total of 2 models for configuration of 1-2 models, then in Interaction page there would be a dropdown auto-populated list in optimal location, showing the 1 non-interactive or if 2 configured then interactive plus able to switch between the 2 models. for mem-lock loading, where it would then be actually switching to the other model if required, upon very next iteration where user clicks to submit input, unloading its current model safely, waiting 2 seconds Printing "Ram Activitiy cooldown for 2 Seconds...", then loading the model the user switched to before pressing Send Input, and so on. Using one-shot mode would instead just use whatever model is selected when it does, as as it should be automatically unloading the model after each iteration ignoring the idle timer auto-unload.
- STT - We could have a STT button in the tools section, enabling the input box to switch to a sample display and a button, the user would click and hold the button to record, and then let go of mouse when they finished recording, and then the wave appear in the box, then AI translate this into words, these words are then shown/editable in the text input box, and the wave record box will hide, but there will be a new button at bottom of text input when STT is enabled, to switch back to the STT Recording box and hide the text box again, so the user can re-record (blanking the previous recorded text upon pressing record). if the user selects STT again to disable it then, the wave box will hide, the text input box will be shown, and the Re-Record button will be hidden. Hmm. Is this the best way to do this? needs a brainstorm, but we want to have annotated conversations with AI in teh conversation log, and ability to edit the given annotations and resend, so one would assume that cancelling here would then make the previous stt annotated recording active text in the editing box, while presented as if the user has not commited to saying that line yet. This would require button for tool on right of User Input box, that then switch the User Input box for 2/3 of vertical from top as a live wave display to show sound input through mic, while underneath a single line of annotation, with dynamic expanding slider bar under.

## Credits
- Qwen-Chatbot was forked from [Chat-Gradio-Gguf](https://github.com/wiseman-timelord/Chat-Gradio-Gguf), a overly-complicated version of Qwen-Chatbot if not on windows and using qwen.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), are used for Inference.
- [Gradio](https://gradio.app) is used for the GUI, AI Programming models are widely trained to handle code for this library.
- [Kokoro TTS](https://github.com/hexgrad/kokoro) is used for the TTS that is faster/smaller than the Corqui TTS.
