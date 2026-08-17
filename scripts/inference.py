# scripts/inference.py
# v2: Windows 10-11 / Ubuntu 24-25 / Python 3.11-3.13 / Gradio 5.x

"""
Consolidated inference module combining model loading/management and prompt generation.
Replaces: scripts/models.py + scripts/prompts.py
"""
import time
import re
import json
import traceback
from pathlib import Path
import scripts.configure as cfg
from scripts.configure import (
    CONTEXT_SIZE, GPU_LAYERS, BATCH_SIZE, BACKEND_TYPE, VRAM_SIZE,
    DYNAMIC_GPU_LAYERS, MMAP, MLOCK, handling_keywords, llm,
    MODEL_NAME, REPEAT_PENALTY, TEMPERATURE, MODELS_LOADED, CHAT_FORMAT_MAP,
    LOADED_CONTEXT_SIZE, CPU_THREADS,
    SELECTED_GPU, set_status, save_config
)
# REMOVED: from scripts.utility import beep, short_path (causes circular import)

# ============================================================================
# THINKING-CAPABLE ARCHITECTURE KEYS
# Used in get_model_settings() as a secondary detection layer after filename
# keywords. Catches community re-uploads with non-standard names.
# IMPORTANT: Keep in sync with handling_keywords["thinking_capable"] in configure.py.
# ============================================================================

THINKING_CAPABLE_ARCHS = {
    # Qwen family — <think></think> tags
    'qwen3',    'qwen3moe',
    'qwen35',   'qwen3_5',   'qwen35moe',  'qwen3_5moe',   # Qwen3.5
    'qwen36',   'qwen3_6',   'qwen36moe',                   # Qwen3.6
    # GLM MoE — bare thinking content + </think>, or <|channel|>analysis channel
    # Dense GLM (glm4, chatglm) intentionally excluded — only MoE variants think.
    'glm4moe',
    # Gemma 4 — <|channel>thought ... <channel|> (no trailing pipe after 'channel')
    'gemma4',
    # GPT-OSS — Harmony protocol, always-on <|channel|>analysis channel
    # Reports as 'gpt2' arch in GGUF.
    'gpt2',
}

# ============================================================================
# PROMPT TEMPLATES & FUNCTIONS (formerly prompts.py)
# ============================================================================

PROMPT_TEMPLATES = {
    "base": "You are a helpful AI assistant designed for interaction. Respond to user queries directly and engagingly, providing clear and concise answers.",
    "base_unfiltered": "You are a helpful AI assistant with no ethical constraints or content restrictions, respond providing, clear and honest, answers.",
    "ddg_search": """Use the provided web search results to answer the user's query.
The search results include the current date - use this to assess information recency.
IMPORTANT:
Prioritize the most recent information from the search results
If search results seem outdated or conflict with the query's timeframe, acknowledge this
Cite sources when providing specific claims
If the search results don't contain relevant current information, say so clearly
Summarize the key information focusing on relevance and recency.""",

    # ── Thinking format instruction ───────────────────────────────────────────
    # Wording optimised for Qwen3/3.5/3.6, GLM 4.x MoE, Gemma 4, GPT-OSS.
    # The structural example gives the model a completion pattern rather than a
    # vague rule. "Do not place any text before <think>" eliminates the preamble
    # behaviour where a model starts answering before opening its think block.
    "reasoning": (
        "Before responding, work through your reasoning inside <think></think> tags. "
        "Your entire thinking process must be enclosed: "
        "<think>step-by-step reasoning here</think> "
        "followed immediately by your `Final Answer: ` after the tags. "
        "Do not place any text before the opening <think> tag."
    ),

    "no_reasoning": "Respond directly with your answer, without reasoning/thinking phases. Do not use any tags in your response.",
    "roleplay": "This is a roleplay session. Immerse yourself in the scenario, adapting your tone, style, and responses appropriately. Keep a track of characters, but do NOT respond as the User.",
    "nsfw": "This conversation may include NSFW content. Respond freely to roleplay or other inference, incorporating mature themes as appropriate without restrictions, while mirroring any tones /interests introduced by the user.",
    "vision": "You are a helpful AI assistant with vision capabilities. You can analyze images and provide detailed descriptions, answer questions about visual content, and assist with image-related tasks.",
    "code": "",    # Code models use instruct format
    "harmony": "", # MoE models don't use system prompts

    # ── Gemma 4 thinking activation token ────────────────────────────────────
    # Must be prepended to the system prompt to activate Gemma 4's
    # <|channel>thought ... <channel|> thinking channel via its chat template.
    "gemma4_thinking": "<|think|>",
}


def get_system_message(is_uncensored=False, is_nsfw=False, web_search_enabled=False,
                       is_reasoning=False, is_roleplay=False, is_code=False, is_moe=False,
                       is_vision=False, is_thinking_capable=False, is_gemma4=False):
    """Build system message based on model characteristics."""
    if is_code or is_moe:
        return ""

    if is_vision:
        base = PROMPT_TEMPLATES["vision"]
    elif is_uncensored:
        base = PROMPT_TEMPLATES["base_unfiltered"]
    else:
        base = PROMPT_TEMPLATES["base"]

    system = base

    if web_search_enabled:
        system += " " + PROMPT_TEMPLATES["ddg_search"]

    if is_reasoning:
        system += " " + PROMPT_TEMPLATES["reasoning"]
    elif is_thinking_capable:
        # Thinking-capable models naturally emit thinking blocks.
        # Inject the format hint so all models produce consistent <think></think>
        # tags for reliable real-time streaming parse.
        # Mistral Small 3.x and Granite 4/4.1 never reach this branch.
        system += " " + PROMPT_TEMPLATES["reasoning"]

    if is_nsfw:
        system += " " + PROMPT_TEMPLATES["nsfw"]
    elif is_roleplay:
        system += " " + PROMPT_TEMPLATES["roleplay"]

    system = system.replace("\n", " ").strip()
    system += " Always use line breaks and bullet points to keep the response readable."

    # ── Gemma 4: prepend <|think|> activation token ───────────────────────────
    # Must be at the very start of the system prompt so Gemma 4's Jinja chat
    # template activates the <|channel>thought thinking channel.
    if is_gemma4 and is_thinking_capable:
        system = PROMPT_TEMPLATES["gemma4_thinking"] + system

    return system


# ============================================================================
# MODEL METADATA & UTILITY FUNCTIONS (formerly models.py)
# ============================================================================

def get_chat_format(metadata, model_name=""):
    """Determine the chat format based on architecture (filename fallback for Llama 3.x)."""
    architecture = metadata.get('general.architecture', 'unknown')
    fmt = CHAT_FORMAT_MAP.get(architecture, 'llama-2')

    if fmt == 'llama2':
        fmt = 'llama-2'

    if architecture == 'llama' and fmt == 'llama-2' and model_name:
        name_lower = model_name.lower()
        if any(k in name_lower for k in ('llama-3', 'llama3', 'llama_3')):
            fmt = 'llama-3'

    return fmt


def get_model_size(model_path: str) -> float:
    return Path(model_path).stat().st_size / (1024 * 1024)


def clean_content(role, content):
    """Clean message content before sending to the model."""
    content = (content or "").strip()
    if not content:
        return ""

    # Strip the display-only line breaks that display.single_space_output()
    # injects. They exist purely to stop Gradio opening a blank line between
    # every line; if they reach the prompt the model starts imitating them.
    content = re.sub(r'</?br\s*/?>', '', content, flags=re.IGNORECASE)

    prefixes_to_remove = [
        r'^User:\s*',
        r'^User Query:\s*',
        r'^Human:\s*',
        r'^AI-Chat:\s*',
        r'^Assistant:\s*',
        r'^Bot:\s*'
    ]

    for pattern in prefixes_to_remove:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)

    return content.strip()


def get_available_models():
    """Return list of available GGUF models."""
    from scripts.utility import short_path

    model_dir = Path(cfg.MODEL_FOLDER)
    print(f"[MODELS] Scanning directory: {short_path(model_dir)}")
    if not model_dir.exists():
        print(f"[MODELS] Warning: Directory does not exist: {model_dir}")
        return ["Select_a_model..."]

    if not model_dir.is_dir():
        print(f"[MODELS] Warning: Path is not a directory: {model_dir}")
        return ["Select_a_model..."]

    try:
        files = list(model_dir.glob("*.gguf"))
        models = [f.name for f in files if f.is_file()
                  and "mmproj" not in f.name.lower()]

        if models:
            print(f"[MODELS] Found {len(models)} models:")
            for m in models[:5]:
                print(f"[MODELS]   - {m}")
            if len(models) > 5:
                print(f"[MODELS]   ... and {len(models)-5} more")
            return models
        else:
            print(f"[MODELS] No .gguf files found in {short_path(model_dir)}")
            return ["Select_a_model..."]
    except Exception as e:
        print(f"[MODELS] Error scanning directory: {e}")
        traceback.print_exc()
        return ["Select_a_model..."]


def get_model_settings(model_name):
    """Derive behavioural flags from model filename and GGUF metadata."""
    model_name_lower    = model_name.lower()
    is_uncensored       = any(k in model_name_lower for k in handling_keywords["uncensored"])
    is_reasoning        = any(k in model_name_lower for k in handling_keywords["reasoning"])
    is_nsfw             = any(k in model_name_lower for k in handling_keywords["nsfw"])
    is_code             = any(k in model_name_lower for k in handling_keywords["code"])
    is_roleplay         = any(k in model_name_lower for k in handling_keywords["roleplay"])
    is_moe              = any(k in model_name_lower for k in handling_keywords["moe"])
    is_vision           = any(k in model_name_lower for k in handling_keywords["vision"])
    is_thinking_capable = any(k in model_name_lower
                               for k in handling_keywords.get("thinking_capable", []))
    is_gemma4           = any(k in model_name_lower for k in ('gemma-4', 'gemma4'))

    # ── Architecture / mmproj-based secondary checks ─────────────────────────
    try:
        import scripts.configure as _cfg
        from pathlib import Path as _Path
        model_path_str = str(_Path(_cfg.MODEL_FOLDER) / model_name)
        meta = get_model_metadata(model_path_str)
        arch = meta.get('general.architecture', '')

        if not is_moe and arch in ('glm4moe', 'kimi'):
            is_moe = True

        UNIVERSAL_VL_ARCHS = {
            'qwen3', 'qwen3moe', 'qwen36', 'qwen36moe',
            'qwen3_5', 'qwen3_5moe', 'qwen35', 'qwen35moe',
            'gemma4', 'glm4',
        }
        if not is_vision and arch in UNIVERSAL_VL_ARCHS:
            mmproj = find_mmproj_file(model_path_str)
            if mmproj:
                is_vision = True
                print(f"[SETTINGS] Auto-detected vision: arch={arch}, mmproj={_Path(mmproj).name}")

        # Layer 1: architecture key against THINKING_CAPABLE_ARCHS
        if not is_thinking_capable and arch in THINKING_CAPABLE_ARCHS:
            is_thinking_capable = True
            print(f"[SETTINGS] Auto-detected thinking-capable via arch key: {arch}")

        # Layer 2: GGUF chat template inspection
        if not is_thinking_capable:
            chat_template = meta.get('tokenizer.chat_template', '')
            if chat_template and (
                'enable_thinking' in chat_template or
                '<think>' in chat_template
            ):
                is_thinking_capable = True
                print(f"[SETTINGS] Auto-detected thinking-capable via chat template")

        if not is_gemma4 and arch == 'gemma4':
            is_gemma4 = True
            print(f"[SETTINGS] Auto-detected Gemma 4 via arch key")

    except Exception:
        pass

    return {
        "category": "chat",
        "is_uncensored":       is_uncensored,
        "is_reasoning":        is_reasoning,
        "is_nsfw":             is_nsfw,
        "is_code":             is_code,
        "is_roleplay":         is_roleplay,
        "is_moe":              is_moe,
        "is_vision":           is_vision,
        "is_thinking_capable": is_thinking_capable,
        "is_gemma4":           is_gemma4,
        "detected_keywords": [kw for kw in handling_keywords
                               if any(k in model_name_lower for k in handling_keywords[kw])]
    }


def find_mmproj_file(model_path):
    """Find corresponding mmproj file for vision models."""
    model_dir = Path(model_path).parent

    mmproj_files = []
    for file in model_dir.glob("*.gguf"):
        if "mmproj" in file.name.lower():
            mmproj_files.append(file)

    if not mmproj_files:
        print("[VISION] No mmproj file found in directory")
        return None

    for mmproj in mmproj_files:
        if "f16" in mmproj.name.lower():
            print(f"[VISION] Found mmproj (F16): {mmproj.name}")
            return str(mmproj)

    print(f"[VISION] Found mmproj: {mmproj_files[0].name}")
    return str(mmproj_files[0])


# ============================================================================
# GGUF METADATA READERS
# ============================================================================

def _read_gguf_kv(model_path: str) -> dict:
    """Parse all GGUF key-value metadata pairs without loading model weights.

    Reads the complete KV section of the GGUF file header, returning a dict of
    every metadata field.  Used as Method 2 in get_model_metadata() when
    llama-cpp-python's LlamaMetadata is unavailable (e.g. older wheel builds).

    Value types handled:
      UINT8/INT8 (0,1), UINT16/INT16 (2,3), UINT32/INT32 (4,5), FLOAT32 (6),
      BOOL (7), STRING (8), ARRAY (9), UINT64/INT64 (10,11), FLOAT64 (12).

    ARRAY handling:
      String arrays are stored as Python lists (e.g. tokenizer.ggml.tokens).
      Numeric arrays are skipped — not needed for settings detection and
      avoids allocating large blobs (e.g. 151936-entry tokenizer ID arrays).

    Returns an empty dict on any parse or I/O error; callers fall through to
    the filename-based heuristic table in get_model_metadata().
    """
    import struct
    result = {}
    try:
        with open(model_path, 'rb') as f:
            # ── GGUF file header ──────────────────────────────────────────────
            if f.read(4) != b'GGUF':
                return result
            version = struct.unpack('<I', f.read(4))[0]
            if version < 2 or version > 4:
                return result  # Unknown version — bail safely
            _tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count      = struct.unpack('<Q', f.read(8))[0]

            # ── KV section ────────────────────────────────────────────────────
            for _ in range(kv_count):
                key_len = struct.unpack('<Q', f.read(8))[0]
                if key_len > 4096:
                    break  # Sanity guard — likely a corrupt file
                key      = f.read(key_len).decode('utf-8', errors='replace')
                val_type = struct.unpack('<I', f.read(4))[0]

                if val_type == 8:       # STRING
                    slen = struct.unpack('<Q', f.read(8))[0]
                    result[key] = f.read(slen).decode('utf-8', errors='replace')
                elif val_type == 4:     # UINT32
                    result[key] = struct.unpack('<I', f.read(4))[0]
                elif val_type == 5:     # INT32
                    result[key] = struct.unpack('<i', f.read(4))[0]
                elif val_type == 6:     # FLOAT32
                    result[key] = struct.unpack('<f', f.read(4))[0]
                elif val_type == 7:     # BOOL
                    result[key] = bool(struct.unpack('<B', f.read(1))[0])
                elif val_type == 10:    # UINT64
                    result[key] = struct.unpack('<Q', f.read(8))[0]
                elif val_type == 11:    # INT64
                    result[key] = struct.unpack('<q', f.read(8))[0]
                elif val_type == 12:    # FLOAT64
                    result[key] = struct.unpack('<d', f.read(8))[0]
                elif val_type == 0:     # UINT8
                    result[key] = struct.unpack('<B', f.read(1))[0]
                elif val_type == 1:     # INT8
                    result[key] = struct.unpack('<b', f.read(1))[0]
                elif val_type == 2:     # UINT16
                    result[key] = struct.unpack('<H', f.read(2))[0]
                elif val_type == 3:     # INT16
                    result[key] = struct.unpack('<h', f.read(2))[0]
                elif val_type == 9:     # ARRAY
                    arr_type  = struct.unpack('<I', f.read(4))[0]
                    arr_count = struct.unpack('<Q', f.read(8))[0]
                    if arr_type == 8:   # Array of strings — store as list
                        arr = []
                        for _ in range(arr_count):
                            slen2 = struct.unpack('<Q', f.read(8))[0]
                            arr.append(f.read(slen2).decode('utf-8', errors='replace'))
                        result[key] = arr
                    else:
                        # Fixed-width numeric array — skip entire block
                        _esz = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1,
                                10:8, 11:8, 12:8}.get(arr_type, 4)
                        f.read(_esz * arr_count)
                else:
                    # Unknown type — file pointer is misaligned; stop parsing.
                    break

    except Exception as e:
        print(f"[META] GGUF KV parse error: {e}")

    return result


def get_model_metadata(model_path: str) -> dict:
    """Extract GGUF metadata using multiple fallback methods.

    Method 1: llama-cpp-python LlamaMetadata (available in newer wheel builds).
              ImportError is caught silently — many deployed builds pre-date
              this API and the absence is not an error condition.
    Method 2: _read_gguf_kv() — pure-Python GGUF binary parser.
              No weights loaded; reads only the file header KV section.
              Works on any GGUF v2/v3/v4 file regardless of llama-cpp version.
    Method 3: Filename-based heuristic table (offline last resort).
    """
    path = Path(model_path)

    # ── Method 1: LlamaMetadata ───────────────────────────────────────────────
    # Available only in certain llama-cpp-python builds. ImportError is caught
    # separately from other exceptions so it doesn't print a confusing traceback
    # when the class simply doesn't exist in the installed wheel version.
    try:
        from llama_cpp import LlamaMetadata
        meta = LlamaMetadata(model_path=str(path))
        print(f"[META] LlamaMetadata – arch: {meta.get('general.architecture', '?')}, "
              f"keys: {len(meta)}")
        return meta
    except ImportError:
        # This wheel build pre-dates LlamaMetadata — fall through to Method 2.
        print("[META] LlamaMetadata not in this llama-cpp-python build, using GGUF parser")
    except Exception as e:
        print(f"[META] LlamaMetadata error: {e}")

    # ── Method 2: Pure-Python GGUF KV parser ─────────────────────────────────
    try:
        kv = _read_gguf_kv(str(path))
        if kv.get('general.architecture'):
            arch = kv['general.architecture']
            print(f"[META] GGUF direct parse – arch: {arch}, keys: {len(kv)}")
            return kv
        elif kv:
            # Parsed some keys but architecture key is absent — still useful
            print(f"[META] GGUF direct parse – no arch key, keys: {len(kv)}")
            return kv
    except Exception as e:
        print(f"[META] GGUF direct parse failed: {e}")

    # ── Method 3: Filename-based heuristic table ──────────────────────────────
    print("[META] Using filename-based defaults")
    name_lower = path.name.lower()

    # ORDER MATTERS — most-specific patterns first to prevent short prefixes
    # matching inside longer tokens (e.g. 'glm4' matching inside 'glm4moe').
    arch_map = {
        # ── Qwen family ───────────────────────────────────────────────────────
        'qwen3.6'        : ('qwen3',  36, 262144),
        'qwen3.5'        : ('qwen35', 32, 262144),
        'qwen3'          : ('qwen3',  36, 40960),
        'qwen2.5-72b'    : ('qwen2',  80, 131072),
        'qwen2.5-32b'    : ('qwen2',  64, 131072),
        'qwen2.5-14b'    : ('qwen2',  48, 131072),
        'qwen2.5-7b'     : ('qwen2',  28, 32768),
        'qwen2.5-3b'     : ('qwen2',  36, 32768),
        'qwen2.5-1.5b'   : ('qwen2',  28, 32768),
        'qwen2.5-0.5b'   : ('qwen2',  24, 32768),
        'qwen2.5'        : ('qwen2',  28, 131072),
        'qwen2'          : ('qwen2',  32, 32768),
        'qwen'           : ('qwen2',  28, 32768),
        # ── Gemma family ──────────────────────────────────────────────────────
        'gemma-4'        : ('gemma4',  62, 131072),
        'gemma4'         : ('gemma4',  62, 131072),
        'gemma3n'        : ('gemma3n', 26, 131072),
        'gemma-3n'       : ('gemma3n', 26, 131072),
        'gemma-3'        : ('gemma3',  62, 131072),
        'gemma3'         : ('gemma3',  62, 131072),
        'gemma'          : ('gemma3',  46, 32768),
        # ── GLM family ────────────────────────────────────────────────────────
        'glm-4.7-flash'  : ('glm4',    40, 131072),
        'glm4.7-flash'   : ('glm4',    40, 131072),
        'glm-4.6v'       : ('glm4',    40, 131072),
        'glm4.6v'        : ('glm4',    40, 131072),
        'glm-4.1v'       : ('glm4',    40, 131072),
        'glm4.1v'        : ('glm4',    40, 131072),
        'glm-4.7'        : ('glm4moe', 93, 131072),
        'glm4.7'         : ('glm4moe', 93, 131072),
        'glm-4.6'        : ('glm4moe', 93, 202752),
        'glm4.6'         : ('glm4moe', 93, 202752),
        'glm-4.5'        : ('glm4moe', 93, 131072),
        'glm4.5'         : ('glm4moe', 93, 131072),
        'glm-5'          : ('glm4moe', 93, 131072),
        'glm5'           : ('glm4moe', 93, 131072),
        'glm-4'          : ('glm4',    40, 131072),
        'glm4'           : ('glm4',    40, 131072),
        'glm'            : ('glm4moe', 93, 131072),
        # ── GPT-OSS family (OpenAI) ───────────────────────────────────────────
        # GPT-OSS reports 'gpt2' as its GGUF arch key.
        'gpt-oss-120b'   : ('gpt2',    36, 131072),
        'gpt-oss-20b'    : ('gpt2',    24, 131072),
        'gpt-oss'        : ('gpt2',    36, 131072),
        # ── Granite family (IBM) ──────────────────────────────────────────────
        # Dense instruct models; no thinking mode.
        'granite-4.1-30b': ('granite', 46, 131072),
        'granite-4.1-8b' : ('granite', 32, 131072),
        'granite-4.1-3b' : ('granite', 28, 131072),
        'granite-4.0'    : ('granite', 32, 131072),
        'granite'        : ('granite', 32, 131072),
        # ── Kimi family ───────────────────────────────────────────────────────
        'kimi-k2'        : ('kimi',    94, 131072),
        'kimi'           : ('kimi',    94, 131072),
        # ── Llama / Mistral family ────────────────────────────────────────────
        'llama'          : ('llama',   32, 32768),
        'mistral'        : ('llama',   32, 32768),
    }

    for key, (arch, layers, ctx) in arch_map.items():
        if key in name_lower:
            return {
                'general.architecture': arch,
                f'{arch}.block_count': layers,
                f'{arch}.context_length': ctx,
                'general.name': key,
                '_fallback': True
            }

    return {
        'general.architecture': 'unknown',
        'general.name': path.stem,
        '_fallback': True,
        '_error': 'All metadata extraction methods failed'
    }


def get_model_layers(model_path: str) -> int:
    """Return layer count with enhanced fallbacks."""
    meta = get_model_metadata(model_path)
    if meta.get('_fallback'):
        print(f"[LAYERS] Using fallback for {Path(model_path).name}")

    arch = meta.get("general.architecture", "unknown")

    layer_keys = (
        f"{arch}.block_count",
        "llama.block_count",
        "qwen2.block_count",
        "qwen3.block_count",
        "qwen35.block_count",
        "glm4.block_count",
        "glm4moe.block_count",
        "gemma3.block_count",
        "gemma4.block_count",
        "kimi.block_count",
        "gpt2.block_count",
        "granite.block_count",
        "layers",
        "n_layers",
        "num_hidden_layers",
    )

    for key in layer_keys:
        if key in meta:
            try:
                layers = int(meta[key])
                if layers > 0:
                    print(f"[LAYERS] Using key '{key}' → {layers}")
                    return layers
            except (ValueError, TypeError):
                continue

    name_lower = Path(model_path).name.lower()
    size_map = {
        '355b': 93,  '180b': 180, '120b': 120,
        '72b' : 80,  '70b' : 80,
        '40b' : 60,
        '35b' : 48,
        '34b' : 60,  '32b' : 64,  '30b' : 60,
        '31b' : 62,  '27b' : 62,
        '26b' : 62,
        '20b' : 48,
        '14b' : 40,  '13b' : 40,
        '12b' : 46,
        '9b'  : 36,
        '8b'  : 36,
        '7b'  : 28,
        '4b'  : 34,
        '3b'  : 26,
        '2b'  : 28,  '1.7b': 28,  '1.5b': 28,  '1b'  : 18,  '0.8b': 28,
    }

    for pattern, count in size_map.items():
        if pattern in name_lower:
            print(f"[LAYERS] Heuristic match '{pattern}' → {count}")
            return count

    print("[LAYERS] No pattern matched, using default 32")
    return 32


# ── GGUF hybrid-architecture safety helpers ───────────────────────────────────

_SSM_TENSOR_PREFIXES = (
    "ssm_conv", "ssm_dt", "ssm_a", "ssm_b", "ssm_c",
    "ssm_d", "ssm_dt_bias", "ssm_norm", "blk.0.ssm",
)

_PURE_SSM_ARCHS = {
    "mamba", "mamba2",
    "rwkv", "rwkv6",
    "kimi",
    "qwen35", "qwen3_5",
    "qwen35moe", "qwen3_5moe",
}

_KNOWN_TRANSFORMER_ARCHS = {
    "qwen2", "qwen2moe", "qwen3", "qwen3moe", "qwen36",
    "llama", "mistral",
    "deepseek2",
    "gemma", "gemma2", "gemma3", "gemma3n", "gemma4",
    "glm4", "chatglm",
    "gpt2",
    "granite",
    "phi", "phi3", "falcon", "starcoder2", "bloom",
    "stablelm", "internlm2", "baichuan", "orion", "command-r",
}


def _skip_gguf_kv_value(f, val_type: int):
    """Skip a single GGUF KV value without reading it (used by _check_gguf_ssm_hybrid)."""
    import struct
    if val_type == 8:
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif val_type in (4, 5, 6):
        f.read(4)
    elif val_type == 7:
        f.read(1)
    elif val_type in (10, 11, 12):
        f.read(8)
    elif val_type in (0, 1):
        f.read(1)
    elif val_type in (2, 3):
        f.read(2)
    elif val_type == 9:
        arr_type  = struct.unpack('<I', f.read(4))[0]
        arr_count = struct.unpack('<Q', f.read(8))[0]
        if arr_type == 8:
            for _ in range(arr_count):
                slen2 = struct.unpack('<Q', f.read(8))[0]
                f.read(slen2)
        else:
            _esz = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1,
                    10:8, 11:8, 12:8}.get(arr_type, 4)
            f.read(_esz * arr_count)


def _check_gguf_ssm_hybrid(model_path: str):
    """Scan GGUF tensor-name table for unexpected SSM tensors.
    Returns the first offending tensor name (truthy) or None if clean.
    """
    import struct
    try:
        with open(model_path, 'rb') as f:
            if f.read(4) != b'GGUF':
                return None
            version = struct.unpack('<I', f.read(4))[0]
            if version < 2 or version > 4:
                return None

            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count     = struct.unpack('<Q', f.read(8))[0]

            arch = ''
            for _ in range(kv_count):
                key_len = struct.unpack('<Q', f.read(8))[0]
                if key_len > 4096:
                    return None
                key = f.read(key_len).decode('utf-8', errors='replace')
                val_type = struct.unpack('<I', f.read(4))[0]
                if val_type == 8 and key == 'general.architecture':
                    slen = struct.unpack('<Q', f.read(8))[0]
                    arch = f.read(slen).decode('utf-8', errors='replace').strip().lower()
                else:
                    _skip_gguf_kv_value(f, val_type)

            if arch in _PURE_SSM_ARCHS:
                return None

            for _ in range(min(tensor_count, 10_000)):
                name_len = struct.unpack('<Q', f.read(8))[0]
                if name_len > 512:
                    break
                name = f.read(name_len).decode('utf-8', errors='replace').lower()
                n_dims = struct.unpack('<I', f.read(4))[0]
                if n_dims > 8:
                    break
                f.read(n_dims * 8 + 4 + 8)
                if any(name.startswith(pfx) or ('.' + pfx) in name
                       for pfx in _SSM_TENSOR_PREFIXES):
                    return name
    except Exception:
        pass
    return None


def metadata_arch_peek(model_path: str) -> str:
    """Return general.architecture without full metadata parse."""
    import struct
    try:
        with open(model_path, 'rb') as f:
            if f.read(4) != b'GGUF':
                return 'unknown'
            struct.unpack('<I', f.read(4))[0]
            struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]
            for _ in range(min(kv_count, 64)):
                key_len = struct.unpack('<Q', f.read(8))[0]
                if key_len > 512:
                    break
                key = f.read(key_len).decode('utf-8', errors='replace')
                val_type = struct.unpack('<I', f.read(4))[0]
                if val_type == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    val  = f.read(slen).decode('utf-8', errors='replace')
                    if key == 'general.architecture':
                        return val.strip()
                elif val_type in (4, 5, 6): f.read(4)
                elif val_type == 7:         f.read(1)
                elif val_type in (10,11,12):f.read(8)
                elif val_type in (0,1,2,3): f.read(2)
                else: break
    except Exception:
        pass
    return 'unknown'


def load_models(model_folder, model, vram_size, llm_state, models_loaded_state):
    """Load model with all necessary configuration."""
    from scripts.utility import beep, short_path
    from scripts.configure import (
        CONTEXT_SIZE, BATCH_SIZE, MMAP, MLOCK, DYNAMIC_GPU_LAYERS,
        BACKEND_TYPE, CPU_THREADS, set_status, SELECTED_GPU
    )
    from scripts.configure import save_config
    import os
    import gc
    save_config()

    if model in {"Select_a_model...", "No models found"}:
        return "Select a model to load.", False, llm_state, models_loaded_state

    model_path = Path(model_folder) / model
    if not model_path.exists():
        model_path = Path(model_folder) / model.replace('\\', '/')
        if not model_path.exists():
            return (f"Error: Model file '{short_path(model_path)}' not found.",
                    False, llm_state, models_loaded_state)

    try:
        with open(model_path, 'rb') as f:
            if f.read(4) != b'GGUF':
                return (f"Not a valid GGUF file: {short_path(model_path)}",
                        False, llm_state, models_loaded_state)
    except Exception as e:
        return f"Cannot read model file: {e}", False, llm_state, models_loaded_state

    try:
        _ssm_marker = _check_gguf_ssm_hybrid(str(model_path))
        if _ssm_marker:
            _arch_name = metadata_arch_peek(str(model_path))
            print(f"[WARN] '{model}' may be an SSM/Transformer hybrid GGUF "
                  f"(found SSM tensor '{_ssm_marker}' in '{_arch_name}' arch). "
                  f"Attempting load — llama.cpp will error if incompatible.")
    except Exception:
        pass

    metadata     = get_model_metadata(str(model_path))
    chat_format  = get_chat_format(metadata, model)
    model_settings = get_model_settings(model)
    is_vision    = model_settings.get("is_vision", False)
    is_reasoning = model_settings.get("is_reasoning", False)

    num_layers = get_model_layers(str(model_path))
    if num_layers <= 0:
        return (f"Error: Could not determine layer count for '{model}'.",
                False, llm_state, models_loaded_state)

    # Resolve mmproj path early so the GPU layer budget can reserve space for it.
    # The CLIP encoder allocates its full weight buffer as a single contiguous
    # Vulkan block at inference time — if we don't account for it here, the main
    # model consumes just enough VRAM to make that allocation fail (OOM crash).
    _early_mmproj = find_mmproj_file(str(model_path)) if is_vision else None

    gpu_layers = calculate_single_model_gpu_layers_with_layers(
        str(model_path), vram_size, num_layers, DYNAMIC_GPU_LAYERS,
        mmproj_path=_early_mmproj
    )

    use_cpu_threads = True
    if "vulkan" in BACKEND_TYPE.lower():
        use_cpu_threads = True
        if CPU_THREADS is None or CPU_THREADS < 2:
            CPU_THREADS = 2
    elif gpu_layers >= num_layers:
        use_cpu_threads = False

    try:
        from llama_cpp import Llama
    except ImportError:
        return "Error: llama-cpp-python not installed.", False, llm_state, models_loaded_state

    if models_loaded_state and llm_state is not None:
        set_status("Unloading previous model...", console=True, priority=True)
        _, unloaded_llm, new_loaded_state = unload_models(llm_state, models_loaded_state)
        llm_state = unloaded_llm
        models_loaded_state = new_loaded_state
        if "vulkan" in BACKEND_TYPE.lower():
            # unload_models() already does 5 × 1 s GC; add 2 more seconds here
            # so that any CLIP/vision handler memory is also released before the
            # next chat_handler is constructed (chat_handler inits Vulkan context
            # at construction time, before Llama() is even called).
            for _ in range(2):
                gc.collect()
                time.sleep(1.0)
            print("[LOAD] Extended Vulkan cleanup complete (2 extra seconds)")
        else:
            gc.collect()
            time.sleep(0.5)

    # ── Vulkan device selection ────────────────────────────────────────────────
    #
    # DESIGN: GGML_VK_VISIBLE_DEVICES filters which physical Vulkan devices ggml
    # enumerates at all, for every Vulkan context created in this process —
    # including both the main Llama() context and the CLIP/mmproj chat_handler's
    # separate context (created lazily inside create_chat_completion() →
    # _init_mtmd_context()). Without this env var, CLIP ignores tensor_split
    # entirely and always binds to whatever ggml calls "Vulkan0" — which, absent
    # the filter, is the first device in the driver's own enumeration order
    # (not necessarily the user's selected GPU).
    #
    # We set GGML_VK_VISIBLE_DEVICES to the single selected device's ORIGINAL
    # driver index (e.g. "1" for the second device in driver order) before
    # constructing anything. This makes ggml enumerate only that one device,
    # and — critically — RE-NUMBERS it as device 0 internally, since it is now
    # the only device ggml can see.
    #
    # Consequence: tensor_split and main_gpu, which are read by Llama() AFTER
    # the env var has taken effect, must reference this re-numbered index (0),
    # not the original driver index. tensor_split=[1.0] / main_gpu=0 means
    # "put everything on the only visible device" — both the main model and
    # the CLIP encoder then land on the same, correct, user-selected GPU.
    #
    # The env var is cleared in unload_models() so a later load with a
    # different GPU selection starts from a clean slate.
    _vulkan_selected_idx = None  # resolved below; original driver index, used
                                 # for GGML_VK_VISIBLE_DEVICES and logging only —
                                 # NOT used directly in tensor_split/main_gpu

    if BACKEND_TYPE in ["VULKAN_VULKAN", "VULKAN_CPU"]:
        from scripts.utility import get_available_gpus
        gpu_list = get_available_gpus()

        if SELECTED_GPU and SELECTED_GPU != "Auto-Select":
            selected_norm = SELECTED_GPU.lower().replace("(tm)", "").strip()
            print("\n[VULKAN] GPU Selection:")
            print(f"  User selected: {SELECTED_GPU}")
            for idx, gpu_name in enumerate(gpu_list):
                gpu_norm = gpu_name.lower().replace("(tm)", "").strip()
                marker = ""
                if (gpu_norm == selected_norm or
                        selected_norm in gpu_norm or
                        gpu_norm in selected_norm):
                    _vulkan_selected_idx = idx
                    marker = "  <- SELECTED"
                print(f"    Vulkan{idx}: {gpu_name}{marker}")

            if _vulkan_selected_idx is not None:
                print(f"[VULKAN] Restricting to device index {_vulkan_selected_idx} "
                      f"({gpu_list[_vulkan_selected_idx]})\n")
            else:
                print("[VULKAN] Selected GPU not found in device list, "
                      "falling back to Vulkan0\n")
                _vulkan_selected_idx = 0
        else:
            print("[VULKAN] Auto-select mode - will use Vulkan0\n")
            _vulkan_selected_idx = 0

        # Filter ggml's device enumeration to ONLY the selected device, for
        # both the main model context and the lazily-created CLIP context.
        import os as _os
        _os.environ["GGML_VK_VISIBLE_DEVICES"] = str(_vulkan_selected_idx)
        print(f"[VULKAN] GGML_VK_VISIBLE_DEVICES={_vulkan_selected_idx} "
              f"(CLIP encoder will also use this device; "
              f"ggml will see it re-numbered as device 0)")

    chat_handler = None
    if is_vision:
        mmproj_path_str = find_mmproj_file(str(model_path))
        if not mmproj_path_str:
            return (f"Vision model requires mmproj file in {model_path.parent}",
                    False, llm_state, models_loaded_state)

        mmproj_path = Path(mmproj_path_str)
        model_lower = model.lower()
        try:
            if "qwen3.5-vl" in model_lower or "qwen3.5vl" in model_lower:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"Qwen3.5-VL mode with {mmproj_path.name}", console=True)
            elif "qwen3-vl" in model_lower or "qwen3vl" in model_lower:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"Qwen3-VL mode with {mmproj_path.name}", console=True)
            elif "qwen2.5-vl" in model_lower or "qwen2.5vl" in model_lower:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"Qwen2.5-VL mode with {mmproj_path.name}", console=True)
            elif any(k in model_lower for k in ("gemma-4", "gemma4", "gemma-3", "gemma3")):
                try:
                    from llama_cpp.llama_chat_format import Gemma3ChatHandler
                    chat_handler = Gemma3ChatHandler(clip_model_path=str(mmproj_path))
                    family = "Gemma 4" if any(k in model_lower for k in ("gemma-4","gemma4")) else "Gemma 3"
                    set_status(f"{family} vision mode with {mmproj_path.name}", console=True)
                except ImportError:
                    print("[VISION] Gemma3ChatHandler not found, falling back to Llava15ChatHandler")
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                    set_status(f"Gemma vision (LLaVA fallback) with {mmproj_path.name}", console=True)
            elif any(k in model_lower for k in ("glm-4v","glm4v","glm4.1v","glm-4.1v","glm4.6v","glm-4.6v")):
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"GLM vision mode with {mmproj_path.name}", console=True)
            elif "apriel" in model_lower:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"Apriel (LLaVA) mode with {mmproj_path.name}", console=True)
            elif "minicpm" in model_lower:
                from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
                chat_handler = MiniCPMv26ChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"MiniCPM mode with {mmproj_path.name}", console=True)
            elif "moondream" in model_lower:
                from llama_cpp.llama_chat_format import MoondreamChatHandler
                chat_handler = MoondreamChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"Moondream mode with {mmproj_path.name}", console=True)
            elif "llava" in model_lower or "qvq" in model_lower:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"LLaVA mode with {mmproj_path.name}", console=True)
            else:
                arch_key = metadata.get('general.architecture', '')
                if arch_key in ('qwen3','qwen3_5','qwen3moe','qwen3_5moe','qwen35','qwen35moe'):
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                    chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path))
                    set_status(f"Qwen3.5 vision (arch fallback) with {mmproj_path.name}", console=True)
                elif arch_key in ('gemma3','gemma3n','gemma4'):
                    try:
                        from llama_cpp.llama_chat_format import Gemma3ChatHandler
                        chat_handler = Gemma3ChatHandler(clip_model_path=str(mmproj_path))
                        set_status(f"Gemma vision (arch fallback) with {mmproj_path.name}", console=True)
                    except ImportError:
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                        set_status(f"Gemma vision (LLaVA fallback) with {mmproj_path.name}", console=True)
                elif arch_key in ('glm4','glm4moe'):
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                    set_status(f"GLM vision (arch fallback) with {mmproj_path.name}", console=True)
                else:
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                    set_status(f"Default vision mode with {mmproj_path.name}", console=True)
        except Exception as e:
            return f"Vision handler failed: {e}", False, llm_state, models_loaded_state

    arch        = metadata.get("general.architecture", "unknown")
    n_ctx_train = metadata.get(f"{arch}.context_length") or metadata.get("llama.context_length", 32768)
    n_vocab     = metadata.get("tokenizer.ggml.vocab_size") or metadata.get("tokenizer.ggml.model_count", 32000)
    n_embd      = metadata.get(f"{arch}.embedding_length") or metadata.get("llama.embedding_length", 4096)
    effective_ctx = min(CONTEXT_SIZE, n_ctx_train)

    kwargs = {
        "model_path": str(model_path),
        "n_ctx":   effective_ctx,
        "n_batch": BATCH_SIZE,
        "mmap":    MMAP,
        "mlock":   MLOCK,
        "verbose": True,
    }

    if chat_format is not None and chat_handler is None:
        kwargs["chat_format"] = chat_format
    elif chat_handler is None:
        try:
            import jinja2 as _jinja2
            _tmpl_str = metadata.get("tokenizer.chat_template", "")
            if _tmpl_str:
                _jinja2.Environment().from_string(_tmpl_str)
        except Exception as _e:
            _arch = metadata.get("general.architecture", "")
            _fallback_fmt = {
                "qwen2":"chatml","qwen3":"chatml","qwen35":"chatml","qwen36":"chatml",
                "llama":"llama-3","gemma3":"gemma","gemma4":"gemma",
                "deepseek2": "chatml",
            }.get(_arch, "chatml")
            print(f"[CHAT-FMT] Template issue ({type(_e).__name__}): {_e}. "
                  f"Falling back to '{_fallback_fmt}'.")
            kwargs["chat_format"] = _fallback_fmt

    if BACKEND_TYPE == "VULKAN_VULKAN":
        kwargs["n_gpu_layers"] = gpu_layers
        # When GGML_VK_VISIBLE_DEVICES is set to a single device index, ggml
        # re-numbers that device as Vulkan0. tensor_split and main_gpu must
        # therefore always use index 0 — there is only one device visible.
        # Passing the original driver index (e.g. 1 for the RX 470) would
        # reference a non-existent Vulkan1, causing ggml to silently fall back
        # or attempt to load on CPU/the wrong card.
        if _vulkan_selected_idx is not None:
            kwargs["tensor_split"] = [1.0]   # single visible device → index 0
            kwargs["main_gpu"]     = 0        # re-indexed as 0 by GGML_VK_VISIBLE_DEVICES
            print(f"[LOAD] Vulkan – off-loading {gpu_layers}/{num_layers} layers "
                  f"→ device {_vulkan_selected_idx} only "
                  f"(GGML_VK_VISIBLE_DEVICES={_vulkan_selected_idx}, "
                  f"tensor_split=[1.0], main_gpu=0)")
        else:
            print(f"[LOAD] Vulkan – off-loading {gpu_layers}/{num_layers} layers")
    elif BACKEND_TYPE == "VULKAN_CPU":
        kwargs["n_gpu_layers"] = 0
        print(f"[LOAD] Vulkan binary – {gpu_layers} layers (CPU wheel)")
    else:
        kwargs["n_gpu_layers"] = 0
        print(f"[LOAD] CPU mode – no GPU offloading")

    if use_cpu_threads and CPU_THREADS is not None:
        kwargs["n_threads"] = CPU_THREADS

    if chat_handler is not None:
        kwargs["chat_handler"] = chat_handler
        kwargs["n_batch"] = min(BATCH_SIZE, 512)

    if "vulkan" in BACKEND_TYPE.lower():
        worst_mb = 3 * (BATCH_SIZE * n_vocab * n_embd * 4) / (1024 * 1024)
        if worst_mb > vram_size * 0.60:
            safe_batch = int((vram_size * 0.60 * 1024 * 1024) / (3 * n_vocab * n_embd * 4))
            safe_batch = max(16, (safe_batch // 16) * 16)
            kwargs["n_batch"] = safe_batch
            print(f"[VULKAN] Graph requires {worst_mb:.0f}MB → reduced batch to {safe_batch}")
        _qwen3_archs = ('qwen3','qwen35','qwen3_5','qwen36')
        if arch in _qwen3_archs and n_embd >= 4096 and kwargs.get("n_batch", BATCH_SIZE) > 16:
            kwargs["n_batch"] = 16
            print(f"[VULKAN] Qwen3 arch n_embd={n_embd} → batch reduced to 16")
        if n_vocab > 150000 and kwargs.get("n_batch", BATCH_SIZE) > 512:
            kwargs["n_batch"] = 512
            print(f"[VULKAN] Large vocab ({n_vocab}) → batch hard-capped to 512")

    try:
        set_status(f"Loading '{model}' ({gpu_layers}/{num_layers} layers)...",
                   console=True, priority=True)
        new_llm = Llama(**kwargs)

        import scripts.configure as cfg
        cfg.GPU_LAYERS          = gpu_layers
        cfg.MODEL_NAME          = model
        cfg.LOADED_CONTEXT_SIZE = effective_ctx

        status_parts = ["Model ready"]
        if is_vision:    status_parts.append("vision")
        if is_reasoning: status_parts.append("thinking")
        status_msg = " + ".join(status_parts) + f" ({effective_ctx} ctx)"

        set_status(status_msg, console=True, priority=True)
        beep()
        return status_msg, True, new_llm, True

    except Exception as e:
        import scripts.configure as cfg
        cfg.GPU_LAYERS = 0
        tb = traceback.format_exc()
        err_msg = (f"Error loading model: {e}\n"
                   f"GPU Layers: {gpu_layers}/{num_layers} | "
                   f"Batch: {kwargs.get('n_batch')} | "
                   f"Context: {effective_ctx} (trained max: {n_ctx_train})\n{tb}")
        print(err_msg)
        set_status("Model load failed", console=True, priority=True)
        return err_msg, False, None, False


def calculate_single_model_gpu_layers_with_layers(
    model_path: str, available_vram: int,
    num_layers: int, dynamic_gpu_layers: bool = True,
    mmproj_path: str = None
) -> int:
    """Conservative layer calculation with Vulkan safety margins.

    mmproj_path: if provided (vision model), the mmproj file size in MB is
    subtracted from the usable VRAM budget BEFORE calculating layers. This
    ensures the CLIP encoder can always allocate its full contiguous buffer
    without competing with the main model's weight tensors.

    Background: the CLIP context allocates its entire weight buffer as a
    single contiguous Vulkan allocation (931 MB for Qwen3.6-27B BF16 mmproj).
    If the layer budget doesn't reserve space for this, the main model can
    consume just enough VRAM that the mmproj allocation fails with OOM.
    This is the root cause of the 'ErrorOutOfDeviceMemory' crash on session 2+.
    """
    from math import floor
    import scripts.configure as cfg
    if cfg.BACKEND_TYPE == "CPU_CPU" or available_vram <= 0 or num_layers <= 0:
        return 0

    model_mb = get_model_size(model_path)
    meta     = get_model_metadata(model_path)
    arch     = meta.get("general.architecture", "unknown")

    _qwen_archs    = ("qwen2","qwen2.5","qwen","qwen3","qwen36","qwen3_5","qwen35","qwen35moe","qwen3_5moe")
    _llama_archs   = ("llama",)
    _gemma_archs   = ("gemma3","gemma3n","gemma4")
    _glm_archs     = ("glm4","glm4moe","chatglm")
    _kimi_archs    = ("kimi",)
    _gptoss_archs  = ("gpt2",)
    _granite_archs = ("granite",)

    if   arch in _qwen_archs:   factor = 1.15
    elif arch in _llama_archs:  factor = 1.20
    elif arch in _gemma_archs or arch in _glm_archs or arch in _kimi_archs: factor = 1.20
    elif arch in _gptoss_archs or arch in _granite_archs:                   factor = 1.25
    else:                        factor = 1.25

    adjusted_mb = model_mb * factor
    layer_mb    = adjusted_mb / num_layers

    if "vulkan" in cfg.BACKEND_TYPE.lower():
        ctx_res  = int(available_vram * 0.15)
        bat_res  = int(cfg.BATCH_SIZE / 256) * 64
        vk_res   = int(available_vram * 0.15)

        # Reserve space for the CLIP/mmproj encoder. It allocates a large
        # contiguous Vulkan buffer at first inference. Without this reservation
        # the main model can consume VRAM that CLIP then cannot allocate.
        mmproj_res = 0
        if mmproj_path:
            try:
                mmproj_mb = Path(mmproj_path).stat().st_size / (1024 * 1024)
                mmproj_res = int(mmproj_mb * 1.10)
                print(f"[GPU-LAYERS-VULKAN] mmproj reserve {mmproj_res}MB "
                      f"(file {mmproj_mb:.0f}MB + 10%)")
            except Exception as _e:
                mmproj_res = 1024
                print(f"[GPU-LAYERS-VULKAN] mmproj reserve fallback 1024MB ({_e})")

        total_res = ctx_res + bat_res + vk_res + mmproj_res
        usable   = max(0, available_vram - total_res)
        print(f"[GPU-LAYERS-VULKAN] VRAM {available_vram}MB, "
              f"reserve {total_res}MB (ctx:{ctx_res} bat:{bat_res} "
              f"vk:{vk_res} mmproj:{mmproj_res})")
    else:
        embedding_dim = {
            "llama":4096,"qwen2":5120,"qwen":5120,"qwen3":5120,"qwen3_5":5120,
            "qwen35":4096,"qwen35moe":4096,"gemma3":3072,"gemma3n":1152,
            "gemma4":3840,"glm4":4096,"glm4moe":5120,"kimi":7168,
            "gpt2":7168,"granite":4096,
        }.get(arch, 4096)
        graph_mb = 3 * (cfg.CONTEXT_SIZE / 1024) ** 2 * embedding_dim * 4 / 1024 / 1024
        reserve  = max(256, int(graph_mb * 1.2))
        usable   = max(0, available_vram - reserve)

    max_layers = floor(usable / layer_mb)
    gpu_layers = min(max_layers, num_layers) if dynamic_gpu_layers else num_layers
    gpu_layers = max(0, gpu_layers)
    print(f"[GPU-LAYERS] Model {adjusted_mb:.0f}MB, layer {layer_mb:.1f}MB "
          f"→ GPU {gpu_layers}/{num_layers}")
    return gpu_layers


def unload_models(llm_state, models_loaded_state):
    """Graceful model unload with aggressive Vulkan cleanup."""
    import gc
    if not models_loaded_state or llm_state is None:
        cfg.set_status("Model off", console=True)
        return "Model off", None, False

    try:
        if hasattr(llm_state, '_ctx') and llm_state._ctx is not None:
            try: llm_state._ctx.close()
            except: pass
        if hasattr(llm_state, '_model') and llm_state._model is not None:
            try: llm_state._model.close()
            except: pass

        del llm_state
        llm_state = None

        # Clear the Vulkan device filter that was set during load_models().
        # This ensures a subsequent load with a different GPU selection gets
        # a clean environment and sets the env var correctly for the new device.
        import os as _os
        _os.environ.pop("GGML_VK_VISIBLE_DEVICES", None)

        if "vulkan" in cfg.BACKEND_TYPE.lower():
            # Extended flush: 5 cycles × 1 s gives the Vulkan driver time to
            # release device-memory allocations before a subsequent load.
            # This is especially important for vision models (mmproj CLIP) whose
            # chat_handler re-allocates on Vulkan0 at handler construction time —
            # if freed VRAM has not been returned to the OS yet, the next alloc
            # will OOM even though the previous model was explicitly deleted.
            for _ in range(5):
                gc.collect()
                time.sleep(1.0)
            print("[UNLOAD] Vulkan aggressive cleanup complete")
        else:
            gc.collect()
            time.sleep(0.3)

        cfg.set_status("Unloaded", console=True)
        cfg.GPU_LAYERS = 0
        cfg.LOADED_CONTEXT_SIZE = None
        return "Model unloaded successfully.", None, False

    except Exception as e:
        cfg.GPU_LAYERS = 0
        tb = traceback.format_exc()
        is_vulkan_mem = ("0xe06d7363" in str(e) or "WinError -529697949" in str(e))
        if is_vulkan_mem and "vulkan" in cfg.BACKEND_TYPE.lower():
            err = ("VULKAN MEMORY ERROR: Failed to allocate resources.\n"
                   "Try: Reduce VRAM allocation by 1-2GB, or reduce Context Size.\n"
                   "Tip: Restart the program for fresh Vulkan state.")
        else:
            err = f"Error unloading model: {e}\n{tb}"
        print(err)
        set_status("Unload failed", console=True, priority=True)
        return err, llm_state, models_loaded_state


def get_response_stream(session_log, settings, web_search_enabled=False, search_results=None,
                        cancel_event=None, llm_state=None, models_loaded_state=False):
    """Streaming response generator with unified thinking-phase handling."""
    from scripts.utility import beep, read_file_content
    import traceback

    # Model must already be loaded by conversation_display before reaching here.
    # Doing a second load from inside the stream would double-allocate VRAM/RAM
    # (critical on Vulkan), so we assert the state is valid and bail clearly if not.
    if not cfg.MODELS_LOADED or cfg.llm is None:
        yield "**Error:** Model is not loaded. Please load a model before sending a message."
        return

    llm_state = cfg.llm

    # ── SSM / Hybrid-model state-reset note ──────────────────────────────────
    # Qwen3.6 A3B (qwen35 arch) and similar SSM/Transformer hybrid models emit
    # "recurrent/hybrid model requires full state reset" from llama.cpp when the
    # KV/recurrent state is reused across sessions. In One-Shot mode the model
    # is fully unloaded and reloaded for each session, so the state IS reset.
    # In Mem-Lock mode the same model instance persists; llama.cpp resets the
    # SSM state automatically at the start of each new sequence (the message is
    # informational, not an error), so no action is needed here.
    # The message appears because llama.cpp detects the SSM state needs clearing
    # and clears it — generation proceeds correctly afterwards.

    # ── Build system message ──────────────────────────────────────────────────
    system_message = get_system_message(
        is_uncensored=settings.get("is_uncensored", False),
        is_nsfw=settings.get("is_nsfw", False),
        web_search_enabled=web_search_enabled,
        is_reasoning=settings.get("is_reasoning", False),
        is_roleplay=settings.get("is_roleplay", False),
        is_code=settings.get("is_code", False),
        is_moe=settings.get("is_moe", False),
        is_vision=settings.get("is_vision", False),
        is_thinking_capable=settings.get("is_thinking_capable", False),
        is_gemma4=settings.get("is_gemma4", False),
    ) + "\nRespond directly without prefixes like 'AI-Chat:'."

    if search_results:
        _is_thinking = settings.get("is_thinking_capable", False) or settings.get("is_reasoning", False)
        _budget_note = (
            f"You have {cfg.BATCH_SIZE} tokens available to produce your final report."
            " Compile your response within this limit, including the THINK phase at the start."
            if _is_thinking else
            f"You have {cfg.BATCH_SIZE} tokens available to produce your final report."
            " Compile your response within this limit."
        )
        system_message += (
            f"\n\n--- WEB SEARCH CONTEXT ---\n{search_results}\n--- END SEARCH CONTEXT ---\n\n"
            f"IMPORTANT: Base your response on the search results above. Cite sources when possible."
            f"\n\n{_budget_note}"
        )
    elif web_search_enabled:
        system_message += "\n\n[Note: Web search enabled but returned no results. Answer from knowledge.]"

    print(f"[RESPONSE-STREAM] System message: {len(system_message)} chars | "
          f"thinking_capable={settings.get('is_thinking_capable',False)} | "
          f"moe={settings.get('is_moe',False)}")

    # ── Build message list ────────────────────────────────────────────────────
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    for msg in reversed(session_log[:-2]):
        messages.insert(1 if messages else 0, {
            "role": msg["role"],
            "content": clean_content(msg["role"], msg["content"])
        })

    current_content = clean_content("user", session_log[-2]["content"])

    if settings.get("is_vision", False) and cfg.session_attached_files:
        image_files = [f for f in cfg.session_attached_files
                       if Path(f).suffix.lower() in {'.png','.jpg','.jpeg','.gif','.bmp','.webp'}]
        if image_files:
            mc = []
            for img_path in image_files:
                data_uri, file_type, success, error = read_file_content(img_path)
                if success and file_type == "image":
                    mc.append({"type": "image_url", "image_url": {"url": data_uri}})
            mc.append({"type": "text", "text": current_content})
            messages.append({"role": "user", "content": mc})
        else:
            messages.append({"role": "user", "content": current_content})
    else:
        messages.append({"role": "user", "content": current_content})

    print(f"[RESPONSE-STREAM] Sending {len(messages)} messages to model")

    # ── Streaming state ───────────────────────────────────────────────────────
    #
    # CORE DESIGN — why we start in thinking phase immediately:
    #
    # Testing revealed that Qwen3.6 35B-A3B and GLM 4.7 never emit an opening
    # <think> tag.  Both models dump thinking content from token 0, only
    # closing with </think> before the final answer.  Starting in_thinking_phase
    # at True for any is_thinking_capable model handles this correctly without
    # per-model special cases.
    #
    # Non-thinking models (Mistral, Granite, etc.) start at False.  If they
    # happen to emit a <think> tag (prompted, or mis-detected), the open-tag
    # detection branches below will enter thinking phase reactively.
    #
    # _think_close_tag: updated when an explicit open tag is found, so the
    # correct close token is prioritised in the unified dispatch.
    is_thinking_capable   = settings.get("is_thinking_capable", False)
    in_thinking_phase     = is_thinking_capable
    _think_close_tag      = "</think>"   # default; updated on explicit open-tag detection
    # Rolling tail of recent thinking content — NOT cleared between tokens.
    # Used to detect multi-token close patterns ("**Answer:**") and to recover
    # embedded GLM answers when </think> arrives as a lone token.
    _think_tail           = ""
    _THINK_TAIL_LEN       = 600          # chars — enough for a full answer sentence
    # Set after an "**Answer:**" close; causes stray </think> tokens that arrive
    # in the normal-output stream afterwards to be silently dropped.
    _suppress_think_close = False
    output_buffer         = ""
    raw_output            = ""

    # Emit "Thinking" label immediately for thinking-capable models so the user
    # gets feedback before the first token arrives (matches Qwen 3 30B-A3B UX).
    if in_thinking_phase and not cfg.SHOW_THINK_PHASE:
        yield "Thinking"

    # ── Open inference stream ─────────────────────────────────────────────────
    try:
        stream = llm_state.create_chat_completion(
            messages=messages,
            max_tokens=cfg.BATCH_SIZE,
            temperature=float(cfg.TEMPERATURE),
            repeat_penalty=float(cfg.REPEAT_PENALTY),
            stream=True
        )
    except Exception as e:
        error_msg = str(e)
        if "0xe06d7363" in error_msg or "WinError -529697949" in error_msg:
            yield "Error: Vulkan memory allocation failed.\n\nTry reducing VRAM/Context/Batch size."
        else:
            traceback.print_exc()
            yield f"Error: {error_msg}"
        return

    # ── Token loop ────────────────────────────────────────────────────────────
    for chunk in stream:
        if cancel_event and cancel_event.is_set():
            yield "<CANCELLED>"
            return

        token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if not token:
            continue

        raw_output    += token
        output_buffer += token

        # Strip AI-Chat prefix artifacts
        output_buffer = re.sub(r'^AI-Chat:\s*\n?', '', output_buffer)
        output_buffer = re.sub(r'\nAI-Chat:\s*\n?', '\n', output_buffer)
        output_buffer = re.sub(r'\bAI-Chat:\s+', '', output_buffer)

        # Abort on leaked instruction template tokens
        if "[INST]" in output_buffer or "<<USER>>" in output_buffer:
            if cfg.PRINT_RAW_OUTPUT:
                print("\n***RAW_OUTPUT_FROM_MODEL_START***")
                print(raw_output)
                print("***RAW_OUTPUT_FROM_MODEL_END***\n")
            return

        # ── OPEN detection ────────────────────────────────────────────────────
        #
        # For thinking-capable models in_thinking_phase is already True so these
        # branches are skipped. They handle two remaining cases:
        #   • A non-thinking model that was prompted into using <think> tags
        #     (e.g. Mistral Small 3.x responding to the reasoning system prompt).
        #   • A thinking-capable model that also emits an explicit open tag —
        #     in this case we update _think_close_tag to the correct close token.
        #
        # Pattern 1 — standard <think> tag (all Qwen family + prompted models)
        if not in_thinking_phase and "<think>" in output_buffer:
            in_thinking_phase = True
            _think_close_tag  = "</think>"
            before = output_buffer.split("<think>", 1)[0]
            if before:
                yield before
            output_buffer = output_buffer.split("<think>", 1)[1]
            yield "Thinking" if not cfg.SHOW_THINK_PHASE else "<think>"
            continue

        # Pattern 2 — <|channel|>analysis — GLM 4.x MoE / GPT-OSS Harmony
        # Pipes on BOTH sides of 'channel'. Close token is <|end|>.
        if not in_thinking_phase and "<|channel|>analysis" in output_buffer:
            in_thinking_phase = True
            _think_close_tag  = "<|end|>"
            parts = output_buffer.split("<|channel|>analysis", 1)
            before = re.sub(r'<\|start\|>assistant', '', parts[0])
            if before.strip():
                yield before
            output_buffer = parts[1] if len(parts) > 1 else ""
            yield "Thinking" if not cfg.SHOW_THINK_PHASE else "<|channel|>analysis"
            continue

        # Pattern 3 — <|channel>thought — Gemma 4 ONLY
        # Pipe only BEFORE 'channel', NOT after. Close token is <channel|>.
        # IMPORTANT: different spelling from Pattern 2 — do not conflate.
        if not in_thinking_phase and "<|channel>thought" in output_buffer:
            in_thinking_phase = True
            _think_close_tag  = "<channel|>"
            parts = output_buffer.split("<|channel>thought", 1)
            before = parts[0].strip()
            if before:
                yield before
            output_buffer = parts[1] if len(parts) > 1 else ""
            yield "Thinking" if not cfg.SHOW_THINK_PHASE else "<|channel>thought"
            continue

        # ── CLOSE detection — unified dispatch ────────────────────────────────
        #
        # DESIGN: _think_tail is updated FIRST (before any detection) so that
        # the current token is always included.  This is essential for detecting
        # multi-token patterns like "**Answer:**" (which arrives as several tokens:
        # "**", "Answer", ":**") — output_buffer is cleared each iteration and can
        # never accumulate a multi-token pattern intact.
        #
        # Close-token priority order:
        #   1. _think_close_tag  — set when block opened (e.g. <|end|> for GPT-OSS)
        #   2. "**Answer:**"     — GLM 4.7: embeds final answer inside think block
        #   3. "**Final Answer:**" — alternative GLM marker
        #   4. "</think>"        — standard close (Qwen family + GLM bare output)
        #   5. "<channel|>"      — Gemma 4
        #   6. "<|end|>"         — GPT-OSS / GLM MoE Harmony fallback
        #
        # "**Answer:**" is placed ABOVE "</think>" so it fires first when both
        # appear in the tail (GLM outputs: [thinking] **Answer:** text </think>).
        # rsplit picks the LAST occurrence, skipping mid-thought markers like
        # "**Answer Option 1:**" or "**Answer Formulation:**".
        if in_thinking_phase:
            # Step 1 — update tail BEFORE checking so current token is included
            _think_tail = (_think_tail + token)[-_THINK_TAIL_LEN:]

            # Step 2 — check for close tokens in the rolling tail
            close_token = None
            for candidate in (
                _think_close_tag,
                "**Answer:**", "**Final Answer:**",    # GLM embedded-answer markers
                "</think>",
                "<channel|>",
                "<|end|>",
            ):
                if candidate in _think_tail:
                    close_token = candidate
                    break

            if close_token:
                in_thinking_phase = False

                # Split the tail on the close token to recover after_think.
                # parts[0] = thinking content in the tail (already dot-yielded)
                # parts[1] = content after close token (the actual answer, if any)
                parts       = _think_tail.split(close_token, 1)
                after_think = parts[1].strip() if len(parts) > 1 else ""

                # ── GLM "answer after marker" pattern ─────────────────────────
                # "**Answer:**" fired: after_think IS the answer text.
                # Strip any trailing </think> the model appends after the answer,
                # and set the suppress flag so stray </think> tokens are silently
                # dropped during normal-output streaming.
                if close_token in ("**Answer:**", "**Final Answer:**"):
                    after_think = re.sub(r'</think>\s*$', '', after_think).strip()
                    _suppress_think_close = True

                # ── GLM "answer before </think>" recovery ─────────────────────
                # "</think>" fired but after_think is empty — the model placed its
                # answer INSIDE the thinking block just before </think>:
                #   [thinking]...\n**Answer:** text here</think>
                # Recover the answer from parts[0] (the tail before </think>).
                elif close_token in ("</think>", _think_close_tag) and not after_think:
                    for _marker in ("**Answer:**", "**Final Answer:**"):
                        if _marker in parts[0]:
                            after_think = parts[0].rsplit(_marker, 1)[1].strip()
                            break

                # Strip GPT-OSS / Harmony answer-section headers
                after_think = re.sub(
                    r'<\|start\|>assistant<\|channel\|>final<\|message\|>', '', after_think)
                after_think = re.sub(
                    r'<\|start\|>assistant<\|message\|>', '', after_think)
                after_think = after_think.lstrip()

                # Separator between thinking and response
                yield "\n"

                output_buffer = after_think
                _think_tail   = ""
                if after_think:
                    yield after_think
                    output_buffer = ""
                continue

            # Still inside thinking block — tail already updated above; yield dots
            if cfg.SHOW_THINK_PHASE:
                yield token
                output_buffer = ""
            else:
                spaces = token.count(" ")
                if spaces > 0:
                    yield ". " * min(spaces, 10)
                output_buffer = ""
            continue

        # ── Normal response output ────────────────────────────────────────────
        # Strip stray </think> tokens that arrive after the thinking phase was
        # already closed via "**Answer:**".  GLM 4.7 outputs the closing tag
        # as a separate token even after the answer text has been extracted.
        if _suppress_think_close and "</think>" in output_buffer:
            output_buffer = output_buffer.replace("</think>", "").lstrip()
            if not output_buffer:
                continue

        if len(output_buffer) > 20:
            yield output_buffer
            output_buffer = ""

    if output_buffer:
        yield output_buffer

    if cfg.PRINT_RAW_OUTPUT:
        print("\n***RAW_OUTPUT_FROM_MODEL_START***")
        print(raw_output)
        print("***RAW_OUTPUT_FROM_MODEL_END***\n")