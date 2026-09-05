# scripts/configure.py
# Qwen-Chatbot: Windows 10 / Python 3.12 / Gradio 5.x / PyQt6 / Qwen GGUF models

import json
import configparser
import threading
import os
from pathlib import Path

import faiss
import numpy as np

# LAZY IMPORTS: langchain imports are deferred to avoid heavy startup cost.
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader

# Settings live in two files, one per settings page. See the PERSISTENCE
# section further down for what each one holds.
CONFIGURATION_PATH = Path("data/configuration.json")
PREFERENCES_PATH   = Path("data/preferences.json")

# =============================================================================
# SYSTEM STATE VARIABLES
# =============================================================================

# System constants (backend, versions, etc.) loaded from data/constants.ini
BACKEND_TYPE = "CPU_CPU"
VULKAN_AVAILABLE = False
LAYER_ALLOCATION_MODE = "SRAM_ONLY"
OS_VERSION = None  # Windows version string
WINDOWS_VERSION = None  # Windows-specific version (10, 11)
EMBEDDING_BACKEND = "sentence_transformers"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LOADED_CONTEXT_SIZE = None
LOADED_BATCH_SIZE = None  # n_batch used when the current model was loaded

# Display/Browser constants
QT_VERSION = 6              # PyQt6 — written by installer; default 6 for v2
DX_FEATURE_LEVEL = 0        # DirectX feature level (0xa000 = 10.0, 0xb000 = 11.0)
GRAPHICS_ACCELERATION = True  # Set by installer
GRADIO_VERSION = "5.x"     # Gradio 5.x — written by installer; default 5.x for v2

# Output Filtering Configuration
#
# One rule set, no preset switch. DEFAULT_FILTER_RULES is what a fresh install
# starts from and what Restore Defaults returns to; the live set is ACTIVE_FILTER
# and it is persisted inside preferences.json with the rest of that page.
#
# Blank-line collapsing is deliberately not done here. A literal find/replace
# pair can only ever shorten a run of newlines by a fixed amount, which is why
# the old rules left a "\n\n" behind and the output stayed double-spaced. That
# job belongs to display.single_space_output(). What remains here is line-ending
# normalisation.
DEFAULT_FILTER_RULES = [
    ("\r\n", "\n"),
    ("\r", "\n"),
]

ACTIVE_FILTER = [tuple(rule) for rule in DEFAULT_FILTER_RULES]
FILTER_MODE = "default"   # informational only: "default" or "custom"

# Configuration variables with defaults
MODEL_FOLDER = "path/to/your/models"
CONTEXT_SIZE = 32768
VRAM_SIZE = 8192
BATCH_SIZE = 1024
TEMPERATURE = 0.66
REPEAT_PENALTY = 1.1
DYNAMIC_GPU_LAYERS = True
MMAP = True
MLOCK = True
LOADING_MODE = "Mem-Lock"   # "Mem-Lock" (mlock=True, keep in RAM) | "One-Shot" (mlock=False, unload after response)
MAX_HISTORY_SLOTS = 12
MAX_ATTACH_SLOTS = 6
SESSION_LOG_HEIGHT = 650
VRAM_OPTIONS = [0, 756, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536]
CTX_OPTIONS = [1024, 2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
BATCH_OPTIONS = [128, 256, 512, 1024, 2048, 4096, 8096]
TEMP_OPTIONS = [0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0]
REPEAT_OPTIONS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
HISTORY_SLOT_OPTIONS = [4, 8, 10, 12, 16]
ATTACH_SLOT_OPTIONS = [2, 4, 6, 8, 10]
SESSION_LOG_HEIGHT_OPTIONS = [250, 450, 550, 600, 625, 650, 700, 800, 1000, 1400]

# General Constants/Variables/Lists/Maps/Arrays
TEMP_DIR = "data/temp"
HISTORY_DIR = "data/history"
SESSION_FILE_FORMAT = "%Y%m%d_%H%M%S"
session_label = ""
current_session_id = None
MODELS_LOADED = False
GENERATION_ACTIVE = False  # True while a response stream is running; blocks unload
AVAILABLE_MODELS = None
SESSION_ACTIVE = False
MODEL_NAME = "Select_a_model..."
GPU_LAYERS = 0
SELECTED_GPU = None

# Pending-new-session management (both Mem-Lock and One-Shot)
# ONE_SHOT_PENDING_NEW_SESSION: set True when user clicks "Start New Session".
# Drives a temporary "..Starting New Session.." placeholder as the first slot
# in the history panel. Cleared (and the placeholder removed) when:
#   • the first complete response is saved and labelled, or
#   • the user clicks an existing session slot (discards the pending new session),
#   • the program exits (SESSION_ACTIVE is False, so nothing is written to disk).
# Name retained for compatibility with existing call sites.
ONE_SHOT_PENDING_NEW_SESSION = False
# ONE_SHOT_LOADING: mutex flag - True while load_models() is executing inside
# conversation_display() for a One-Shot new-session request. Prevents a second
# concurrent call from triggering a double-load that exhausts VRAM.
ONE_SHOT_LOADING = False
USE_PYTHON_BINDINGS = True
DATA_DIR = None  # Will be set by launcher.py
llm = None
LLAMA_CLI_PATH = None  # Will be set from constants.ini
LLAMA_BIN_PATH = None  # Will be set from constants.ini
LLAMA_WHEEL_VERSION = None  # Will be set from constants.ini (llama-cpp-python wheel version)
global_status = None
_status_lock = None  # Tracks which operation has status priority
_status_lock_message = ""  # Message to restore when lock releases
PRINT_RAW_OUTPUT = False
SHOW_THINK_PHASE = False
BLEEP_ON_EVENTS = False
USER_INPUT_MAX_LINES = 10  # Recalculated by display.py based on SESSION_LOG_HEIGHT

# CPU Configuration
CPU_THREADS = None  # Will be auto-detected
CPU_PHYSICAL_CORES = 1
CPU_LOGICAL_CORES = 1
SELECTED_CPU = "Auto-Select"

# =============================================================================
# Sound Hardware Configuration (shared by Bleep and TTS)
# =============================================================================
SOUND_OUTPUT_DEVICE = "Default Sound Device"
SOUND_SAMPLE_RATE = 44100
SOUND_SAMPLE_RATE_OPTIONS = [44100, 48000]

# =============================================================================
# TTS (Text-to-Speech) Configuration
# =============================================================================
TTS_ENABLED = False
TTS_ENGINE = "none"
TTS_AUDIO_BACKEND = "none"
TTS_VOICE = None
TTS_VOICE_NAME = None
MAX_TTS_LENGTH = 4500
KOKORO_VOICE = "af_heart"
KOKORO_LANG_CODE = "a"
# New pack-related variables
TTS_PACK = 1
TTS_ENABLED_VOICES = []    # list of voice IDs (strings)
TTS_DEFAULT_VOICE_ID = None
TTS_DEFAULT_VOICE_NAME = None

# =============================================================================
# Per-Message TTS Button State (4-phase cycle: play -> generating -> playing -> idle)
# =============================================================================
TTS_PHASE = "idle"            # "idle" | "generating" | "playing"
TTS_CURRENT_MSG_IDX = None    # Which bot message index is active (0-based)
TTS_BUSY = False              # Whether any TTS operation is in progress
_tts_state_lock = threading.Lock()  # Thread-safety for TTS state globals

# Arrays
session_attached_files = []
session_vector_files = []

# Application icon. Used for the Qt window icon, the Windows taskbar button and
# the Gradio favicon, so the three stay in step from one definition.
APP_ICON_PATH = Path(__file__).resolve().parent.parent / "images" / "Icon_Terminal.ico"
# Explicit AppUserModelID: without one, Windows groups the taskbar button under
# the host python.exe and shows the interpreter's icon instead of the window's.
APP_USER_MODEL_ID = "WiseManTimeLord.QwenWindowsGguf"


def get_app_icon_path():
    """Return the icon path as a string, or None when the file is absent."""
    return str(APP_ICON_PATH) if APP_ICON_PATH.is_file() else None


# UI Constants/Variables
USER_COLOR = "#ffffff"
THINK_COLOR = "#c8a2c8"
RESPONSE_COLOR = "#add8e6"
SEPARATOR = "=" * 40
MID_SEPARATOR = "-" * 30
ALLOWED_EXTENSIONS = {"bat", "py", "ps1", "txt", "json", "yaml", "psd1", "xaml",
                      "png", "jpg", "jpeg", "gif", "bmp", "webp"}
MMPROJ_EXTENSIONS = ["-mmproj-", "mmproj"]
MAX_POSSIBLE_HISTORY_SLOTS = 16
MAX_POSSIBLE_ATTACH_SLOTS = 10
demo = None

# RAG CONSTANTS
RAG_CHUNK_SIZE_DIVIDER = 6
RAG_CHUNK_OVERLAP_DIVIDER = 24
LARGE_INPUT_THRESHOLD = 0.4
RAG_RETRIEVAL_CHUNKS = 8
CONTEXT_ALLOCATION_RATIOS = {
    "system": 0.1,
    "history": 0.3,
    "current": 0.6
}

# Status text entries
STATUS_MESSAGES = {
    "model_loading": "Loading model...",
    "model_loaded": "Model loaded successfully",
    "model_unloading": "Unloading model...",
    "model_unloaded": "Model unloaded successfully",
    "vram_calc": "Calculating layers...",
    "rag_process": "Analyzing documents...",
    "session_restore": "Restoring session...",
    "config_saved": "Settings saved",
    "docs_processed": "Documents ready",
    "generating_response": "Generating response...",
    "response_generated": "Response generated",
    "error": "An error occurred",
    "tts_speaking": "Speaking...",
    "tts_stopped": "Speech stopped"
}

# =============================================================================
# CHAT FORMAT MAP
# Maps GGUF architecture keys to llama-cpp-python chat format strings.
# None = use the chat template embedded in the GGUF (preferred for newer models).
# =============================================================================
CHAT_FORMAT_MAP = {
    # ── Qwen family ───────────────────────────────────────────────────────────
    # All Qwen3+ variants ship with embedded Jinja chat templates that
    # handle the enable_thinking flag and <think> tags natively.
    'qwen2'      : None,
    'qwen3'      : None,
    'qwen36'     : None,
    'qwen3moe'   : None,
    'qwen3_5'    : None,
    'qwen35'     : None,
    'qwen3_5moe' : None,
    'qwen35moe'  : None,
}

# =============================================================================
# HANDLING KEYWORDS
# Derive behavioural flags from model filename tokens.
# Keys map to lists of lowercase substrings; a match sets the corresponding flag.
# =============================================================================
handling_keywords = {
    "code": ["code", "program", "dev", "copilot", "python", "powershell"],

    "uncensored": [
        "uncensored", "unfiltered", "unbiased", "unlocked", "abliterat",
        "heretic", "deeprefusal", "deep-refusal", "claudeopus", "claude-opus",
    ],

    "reasoning": [
        "reason", "r1", "think", "thinking", "z1",
    ],

    "nsfw": ["nsfw", "adult", "mature", "explicit", "lewd"],
    "roleplay": ["rpg", "role", "adventure"],

    "moe": [
        "moe", "a3b", "a22b", "a14b",
    ],

    "vision": [
        "vision", "qvq",
        "qwen3.5-vl", "qwen3.5vl", "qwen3-vl", "qwen3vl", "qwen2.5-vl",
    ],

    # ── thinking_capable ──────────────────────────────────────────────────────
    # Models that natively emit structured thinking blocks during generation.
    # The flag causes get_system_message() to inject a thinking-format hint
    # into the system prompt.
    "thinking_capable": [
        # ── Qwen family ───────────────────────────────────────────────────────
        # Qwen3: all sizes always think.
        # Qwen3.5/3.6 Small (0.8B–9B): thinking opt-in via chat template, but
        # can still be triggered by system-prompt instruction; included so the
        # hint is always injected for consistent output format.
        "qwen3",
        "qwen3.5", "qwen35", "qwen3_5",    # Qwen3.5 — all size variants
        "qwen3.6", "qwen36", "qwen3_6",    # Qwen3.6 — all size variants
    ],
}

# =============================================================================
# CONTEXT INJECTOR CLASS (RAG)
# =============================================================================

class ContextInjector:
    """
    Universal RAG with support for both file attachments AND large pasted inputs.
    Provides unlimited context through intelligent chunking and retrieval.
    Uses sentence-transformers for embeddings.
    """
    def __init__(self):
        self.embedding = None
        self.file_index = None
        self.temp_index = None
        self.file_chunks = []
        self.temp_chunks = []
        self._model_load_attempted = False
        self._embedding_dim = None
        self._model_name = None

    def _ensure_embedding_model(self):
        """Initialize embedding model on first use with proper cache path (fully offline)."""
        if self._model_load_attempted and self.embedding is not None:
            return

        if self.embedding is None:
            self._model_load_attempted = False

        self._model_load_attempted = True

        cache_dir = Path(__file__).parent.parent / "data" / "embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir.absolute())
        os.environ["HF_HOME"] = str(cache_dir.parent.absolute())
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir.absolute())
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

        try:
            from sentence_transformers import SentenceTransformer

            model_name = EMBEDDING_MODEL_NAME
            self._model_name = model_name

            # HuggingFace stores models as  models--ORG--NAME/  (double-dash separated).
            # The old check used model_name.replace("/", "_") which never matched the
            # actual directory, causing the "Downloading/loading" branch to always run
            # even when the model was fully cached from the installer.
            hf_cache_dir = cache_dir / ("models--" + model_name.replace("/", "--"))

            print(f"[RAG] Loading embedding model: {model_name}")

            if hf_cache_dir.exists():
                print(f"[RAG] Loading from cache: {hf_cache_dir}")
            else:
                print(f"[RAG] Cache not found at {hf_cache_dir} — loading (offline)")
            self.embedding = SentenceTransformer(model_name, cache_folder=str(cache_dir))

            if self.embedding is None:
                raise RuntimeError("Model loading returned None")

            test_embedding = self.embedding.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
            self._embedding_dim = test_embedding.shape[1]

            print(f"[RAG] Embedding model loaded successfully (dim={self._embedding_dim})")

        except Exception as e:
            print(f"[RAG] Failed to load embedding model: {e}")
            print(f"[RAG] Tried model: {EMBEDDING_MODEL_NAME}")
            self.embedding = None
            self._embedding_dim = None
            self._model_load_attempted = False

    def _embed_texts(self, texts, batch_size=32):
        """Create embeddings for a list of texts using sentence-transformers."""
        if self.embedding is None:
            return None
        try:
            import numpy as np
            embeddings = self.embedding.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            print(f"[RAG] Embedding error: {e}")
            if "out of memory" in str(e).lower() or "unable to allocate" in str(e).lower():
                print(f"[RAG] Memory error - consider using smaller embedding model (current: {EMBEDDING_MODEL_NAME})")
            return None

    def set_session_vectorstore(self, file_paths):
        """Create FAISS index from attached files for RAG retrieval."""
        # LAZY IMPORT: Import here to avoid heavy startup cost
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._ensure_embedding_model()

        if self.embedding is None:
            print("[RAG] Cannot create vectorstore - embedding model unavailable")
            return

        if not file_paths:
            self.file_index = None
            self.file_chunks = []
            return

        effective_ctx = LOADED_CONTEXT_SIZE or CONTEXT_SIZE
        chunk_size = effective_ctx // RAG_CHUNK_SIZE_DIVIDER
        chunk_overlap = effective_ctx // RAG_CHUNK_OVERLAP_DIVIDER

        texts = []
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len
                    )
                    chunks = splitter.split_text(content)
                    texts.extend(chunks)
            except Exception as e:
                print(f"[RAG] Error loading {path}: {e}")

        if not texts:
            self.file_index = None
            self.file_chunks = []
            return

        embeddings = self._embed_texts(texts, batch_size=16)

        if embeddings is None:
            print("[RAG] Failed to create embeddings - index not created")
            self.file_index = None
            self.file_chunks = []
            return

        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]

        if embeddings.shape[1] != self._embedding_dim:
            print(f"[RAG] Dimension mismatch! Expected {self._embedding_dim}, got {embeddings.shape[1]}")
            self.file_index = None
            return

        self.file_index = faiss.IndexFlatIP(self._embedding_dim)
        faiss.normalize_L2(embeddings)
        self.file_index.add(embeddings)
        self.file_chunks = texts
        print(f"[RAG] Ingested {len(texts)} chunks from {len(file_paths)} files (dim={self._embedding_dim})")

    def add_temporary_input(self, large_input_text):
        """Chunk and index large pasted user input for RAG retrieval."""
        # LAZY IMPORT: Import here to avoid heavy startup cost
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        if not large_input_text or not large_input_text.strip():
            return

        self._ensure_embedding_model()

        if self.embedding is None:
            print("[RAG] Cannot chunk temporary input - embedding model unavailable")
            return

        effective_ctx = LOADED_CONTEXT_SIZE or CONTEXT_SIZE
        chunk_size = effective_ctx // RAG_CHUNK_SIZE_DIVIDER
        chunk_overlap = effective_ctx // RAG_CHUNK_OVERLAP_DIVIDER

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(large_input_text)

        if not chunks:
            return

        embeddings = self._embed_texts(chunks, batch_size=8)

        if embeddings is None:
            print("[RAG] Failed to create temporary embeddings")
            return

        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]

        if embeddings.shape[1] != self._embedding_dim:
            print(f"[RAG] Temp dimension mismatch!")
            return

        self.temp_index = faiss.IndexFlatIP(self._embedding_dim)
        faiss.normalize_L2(embeddings)
        self.temp_index.add(embeddings)
        self.temp_chunks = chunks
        print(f"[RAG] Indexed {len(chunks)} temporary input chunks (dim={self._embedding_dim})")

    def clear_temporary_input(self):
        """Clear any pending temporary RAG input (called after each response cycle)."""
        self.temp_index = None
        self.temp_chunks = []

    def retrieve_context(self, query, k=None):
        """Retrieve relevant context from both file and temporary indexes."""
        if k is None:
            k = RAG_RETRIEVAL_CHUNKS

        self._ensure_embedding_model()

        if self.embedding is None:
            return None

        if self.file_index is None and self.temp_index is None:
            return None

        query_embedding = self._embed_texts([query], batch_size=1)
        if query_embedding is None:
            return None

        faiss.normalize_L2(query_embedding)

        all_results = []
        seen_chunks = set()

        # Search file index
        if self.file_index is not None and self.file_chunks:
            try:
                actual_k = min(k, len(self.file_chunks))
                scores, idxs = self.file_index.search(query_embedding, actual_k)
                file_results = []
                for i in idxs[0]:
                    if i < len(self.file_chunks) and len(file_results) < k:
                        chunk = self.file_chunks[i]
                        chunk_key = chunk[:50].strip()
                        if chunk_key not in seen_chunks:
                            seen_chunks.add(chunk_key)
                            file_results.append(("FILE", chunk))
                all_results.extend(file_results)
                print(f"[RAG] Retrieved {len(file_results)} file chunks")
            except Exception as e:
                print(f"[RAG] Error searching file chunks: {e}")

        # Search temporary index
        if self.temp_index is not None and self.temp_chunks:
            try:
                actual_k = min(k, len(self.temp_chunks))
                scores, idxs = self.temp_index.search(query_embedding, actual_k)
                temp_results = []
                for i in idxs[0]:
                    if i < len(self.temp_chunks) and len(temp_results) < k:
                        chunk = self.temp_chunks[i]
                        chunk_key = chunk[:50].strip()
                        if chunk_key not in seen_chunks:
                            seen_chunks.add(chunk_key)
                            temp_results.append(("TEMP", chunk))
                all_results.extend(temp_results)
                print(f"[RAG-TEMP] Retrieved {len(temp_results)} temporary input chunks")
            except Exception as e:
                print(f"[RAG-TEMP] Error searching temp chunks: {e}")

        if not all_results:
            return None

        formatted_chunks = []
        for source, chunk in all_results:
            if source == "FILE":
                formatted_chunks.append(f"[From Attached Files]\n{chunk}")
            else:
                formatted_chunks.append(f"[From Your Input]\n{chunk}")

        return "\n\n".join(formatted_chunks)


context_injector = ContextInjector()

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def load_system_ini():
    """Load system constants from constants.ini (created by installer)."""
    ini_path = Path("data/constants.ini")
    if not ini_path.exists():
        raise RuntimeError(
            f"System configuration file not found: {ini_path}\n"
            "Re-run the installer to generate constants.ini."
        )
    try:
        config = configparser.ConfigParser()
        config.read(ini_path, encoding='utf-8')

        if 'system' not in config:
            raise RuntimeError("constants.ini missing [system] section")

        system = config['system']

        global BACKEND_TYPE, VULKAN_AVAILABLE, EMBEDDING_MODEL_NAME
        global EMBEDDING_BACKEND, GRADIO_VERSION, LLAMA_CLI_PATH, LLAMA_BIN_PATH
        global OS_VERSION, WINDOWS_VERSION, KOKORO_LANG_CODE
        global GRAPHICS_ACCELERATION, QT_VERSION, DX_FEATURE_LEVEL
        global LLAMA_WHEEL_VERSION
        global TTS_PACK, TTS_ENABLED_VOICES, TTS_DEFAULT_VOICE_ID, TTS_DEFAULT_VOICE_NAME

        BACKEND_TYPE = system.get('backend_type', 'CPU_CPU')
        VULKAN_AVAILABLE = system.getboolean('vulkan_available', False)
        GRAPHICS_ACCELERATION = system.getboolean('browser_acceleration', True)

        # v2: default Qt version is 6 (PyQt6)
        QT_VERSION = system.getint('qt_version', 6)
        DX_FEATURE_LEVEL = system.getint('dx_feature_level', 0xb000)

        EMBEDDING_MODEL_NAME = system.get('embedding_model', 'BAAI/bge-small-en-v1.5')
        EMBEDDING_BACKEND = system.get('embedding_backend', 'sentence_transformers')
        GRADIO_VERSION = system.get('gradio_version', '5.x')
        LLAMA_CLI_PATH = system.get('llama_cli_path', None)
        LLAMA_BIN_PATH = system.get('llama_bin_path', None)
        LLAMA_WHEEL_VERSION = system.get('llama_wheel_version', None)

        print(f"[INI] Backend: {BACKEND_TYPE}")
        print(f"[INI] Vulkan: {VULKAN_AVAILABLE}")
        print(f"[INI] Graphics Acceleration: {GRAPHICS_ACCELERATION}")
        print(f"[INI] Qt Version: {QT_VERSION} (v{QT_VERSION})")
        print(f"[INI] DX Feature Level: 0x{DX_FEATURE_LEVEL:04x}")
        print(f"[INI] Embedding Model: {EMBEDDING_MODEL_NAME}")
        print(f"[INI] Gradio Version: {GRADIO_VERSION}")

        # Single source of truth: os_version
        OS_VERSION = system.get('os_version', 'unknown')

        # Extract raw version number for logic if needed
        WINDOWS_VERSION = OS_VERSION.replace("Windows ", "") if OS_VERSION.startswith("Windows ") else "unknown"

        print(f"[INI] OS Version: {OS_VERSION}")
        print(f"[INI] Windows Version: {WINDOWS_VERSION}")

        # Load TTS configuration from [tts] section (written by installer).
        # These are installer decisions and are authoritative — the JSON never
        # overrides them.
        if 'tts' in config:
            tts_section          = config['tts']
            TTS_PACK             = tts_section.getint('tts_pack', 1)
            TTS_DEFAULT_VOICE_ID = tts_section.get('tts_default_voice_id', 'bm_george')
            TTS_DEFAULT_VOICE_NAME = tts_section.get('tts_default_voice_name', 'George — British Male')
            enabled_str          = tts_section.get('tts_enabled_voices', '')
            TTS_ENABLED_VOICES   = [v.strip() for v in enabled_str.split(',') if v.strip()]
            # KOKORO_LANG_CODE: derive from default voice prefix (am_/af_ -> 'a', bm_/bf_ -> 'b')
            KOKORO_LANG_CODE     = 'b' if TTS_DEFAULT_VOICE_ID.startswith('b') else 'a'
            print(f"[INI] TTS pack {TTS_PACK}: {len(TTS_ENABLED_VOICES)} voice(s) enabled, "
                  f"default={TTS_DEFAULT_VOICE_ID} (lang={KOKORO_LANG_CODE})")
        else:
            print("[INI] No [tts] section — TTS will be disabled")
            TTS_PACK             = 1
            TTS_DEFAULT_VOICE_ID = None
            TTS_DEFAULT_VOICE_NAME = None
            TTS_ENABLED_VOICES   = []
            KOKORO_LANG_CODE     = 'a' 

        return True

    except Exception as e:
        raise RuntimeError(f"Cannot read constants.ini: {e}") from e


# =============================================================================
# PERSISTENCE
# =============================================================================
# Two settings files, one per settings page:
#
#   data/configuration.json  <- the Configuration page (hardware, TTS, model)
#   data/preferences.json    <- the Preferences page (interface, output, filter)
#
# The two dictionaries below are the single source of truth for what each file
# holds and for what "Restore Defaults" resets to. installer.py carries its own
# copy of both (see create_config_jsons) because it runs on the system Python
# before the venv exists and so cannot import this module. Keep them in step.

CONFIGURATION_DEFAULTS = {
    # Model
    "model_dir": "models",
    "model_name": "Select_a_model...",
    "context_size": 32768,
    "n_batch": 1024,
    "temperature": 0.66,
    "repeat_penalty": 1.1,
    "mmap": True,
    "dynamic_gpu_layers": True,
    "use_python_bindings": True,
    # Hardware
    "selected_cpu": "Auto-Select",
    "cpu_threads": None,
    "selected_gpu": None,
    "vram_size": 8192,
    "loading_mode": "Mem-Lock",
    "layer_allocation_mode": "SRAM_ONLY",
    # TTS and audio
    "tts_enabled": False,
    "tts_voice": None,
    "tts_voice_name": None,
    "max_tts_length": 4500,
    "sound_output_device": "Default Sound Device",
    "sound_sample_rate": 44100,
}

PREFERENCES_DEFAULTS = {
    # Program options
    "session_log_height": 650,
    "max_attach_slots": 6,
    "max_history_slots": 12,
    # Output options
    "show_think_phase": False,
    "bleep_on_events": False,
    "print_raw_output": False,
    # Filter settings, as a list of [find, replace] pairs
    "filter_rules": [list(rule) for rule in DEFAULT_FILTER_RULES],
}


def _read_settings(path: Path, section: str, required: bool) -> dict:
    """Read one section out of a settings file. Returns {} when absent."""
    if not path.exists():
        if required:
            raise RuntimeError(f"Settings file not found: {path}\nRe-run the installer.")
        print(f"[CONFIG] {path} not found — using defaults")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            if required:
                raise RuntimeError(f"Settings file {path} is empty")
            print(f"[CONFIG] {path} is empty — using defaults")
            return {}
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Cannot parse settings file {path}: {e}") from e
    except OSError as e:
        raise RuntimeError(f"Cannot read settings file {path}: {e}") from e

    if not isinstance(data, dict):
        raise RuntimeError(f"Settings file {path} does not contain a JSON object")

    values = data.get(section, {})
    if not isinstance(values, dict):
        raise RuntimeError(f"Settings file {path} has an invalid '{section}' section")
    return values


def _write_settings(path: Path, section: str, values: dict):
    """Write one section to a settings file, replacing whatever was there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({section: values}, f, indent=4, ensure_ascii=False)


def _apply_configuration(values: dict):
    """Copy a configuration mapping into the module globals."""
    global MODEL_FOLDER, MODEL_NAME, CONTEXT_SIZE, BATCH_SIZE, TEMPERATURE
    global REPEAT_PENALTY, MMAP, MLOCK, LOADING_MODE, DYNAMIC_GPU_LAYERS
    global USE_PYTHON_BINDINGS, SELECTED_CPU, CPU_THREADS, SELECTED_GPU
    global VRAM_SIZE, LAYER_ALLOCATION_MODE, TTS_ENABLED, TTS_VOICE
    global TTS_VOICE_NAME, MAX_TTS_LENGTH, SOUND_OUTPUT_DEVICE, SOUND_SAMPLE_RATE

    def get(key):
        return values.get(key, CONFIGURATION_DEFAULTS[key])

    # Model
    MODEL_FOLDER        = get("model_dir")
    MODEL_NAME          = get("model_name")
    CONTEXT_SIZE        = get("context_size")
    BATCH_SIZE          = get("n_batch")
    TEMPERATURE         = get("temperature")
    REPEAT_PENALTY      = get("repeat_penalty")
    MMAP                = get("mmap")
    DYNAMIC_GPU_LAYERS  = get("dynamic_gpu_layers")
    USE_PYTHON_BINDINGS = get("use_python_bindings")

    # Hardware
    SELECTED_CPU          = get("selected_cpu")
    CPU_THREADS           = get("cpu_threads")
    SELECTED_GPU          = get("selected_gpu")
    VRAM_SIZE             = get("vram_size")
    LOADING_MODE          = get("loading_mode")
    LAYER_ALLOCATION_MODE = get("layer_allocation_mode")
    # MLOCK is derived, not stored: LOADING_MODE is the canonical source.
    MLOCK = (LOADING_MODE == "Mem-Lock")

    # TTS and audio. Pack, enabled voice list and default voice are read from
    # constants.ini by load_system_ini() and never round-tripped through JSON.
    TTS_ENABLED       = get("tts_enabled")
    TTS_VOICE         = get("tts_voice")
    TTS_VOICE_NAME    = get("tts_voice_name")
    MAX_TTS_LENGTH    = get("max_tts_length")
    SOUND_SAMPLE_RATE = get("sound_sample_rate")
    SOUND_OUTPUT_DEVICE = "Default Sound Device"


def _apply_preferences(values: dict):
    """Copy a preferences mapping into the module globals."""
    global SESSION_LOG_HEIGHT, MAX_ATTACH_SLOTS, MAX_HISTORY_SLOTS
    global SHOW_THINK_PHASE, BLEEP_ON_EVENTS, PRINT_RAW_OUTPUT
    global ACTIVE_FILTER, FILTER_MODE

    def get(key):
        return values.get(key, PREFERENCES_DEFAULTS[key])

    SESSION_LOG_HEIGHT = get("session_log_height")
    MAX_ATTACH_SLOTS   = get("max_attach_slots")
    MAX_HISTORY_SLOTS  = get("max_history_slots")
    SHOW_THINK_PHASE   = get("show_think_phase")
    BLEEP_ON_EVENTS    = get("bleep_on_events")
    PRINT_RAW_OUTPUT   = get("print_raw_output")

    rules = get("filter_rules") or []
    ACTIVE_FILTER = [
        (str(pair[0]), str(pair[1])) for pair in rules
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ]
    FILTER_MODE = "default" if ACTIVE_FILTER == list(DEFAULT_FILTER_RULES) else "custom"


def _configuration_values() -> dict:
    """Build the configuration mapping from the current globals."""
    return {
        "model_dir": MODEL_FOLDER,
        "model_name": MODEL_NAME,
        "context_size": CONTEXT_SIZE,
        "n_batch": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "repeat_penalty": REPEAT_PENALTY,
        "mmap": MMAP,
        "dynamic_gpu_layers": DYNAMIC_GPU_LAYERS,
        "use_python_bindings": USE_PYTHON_BINDINGS,
        "selected_cpu": SELECTED_CPU or "Auto-Select",
        "cpu_threads": CPU_THREADS,
        "selected_gpu": SELECTED_GPU,
        "vram_size": VRAM_SIZE,
        "loading_mode": LOADING_MODE,
        "layer_allocation_mode": LAYER_ALLOCATION_MODE,
        "tts_enabled": TTS_ENABLED,
        "tts_voice": TTS_VOICE,
        "tts_voice_name": TTS_VOICE_NAME,
        "max_tts_length": MAX_TTS_LENGTH,
        "sound_output_device": SOUND_OUTPUT_DEVICE,
        "sound_sample_rate": SOUND_SAMPLE_RATE,
    }


def _preferences_values() -> dict:
    """Build the preferences mapping from the current globals."""
    return {
        "session_log_height": SESSION_LOG_HEIGHT,
        "max_attach_slots": MAX_ATTACH_SLOTS,
        "max_history_slots": MAX_HISTORY_SLOTS,
        "show_think_phase": SHOW_THINK_PHASE,
        "bleep_on_events": BLEEP_ON_EVENTS,
        "print_raw_output": PRINT_RAW_OUTPUT,
        "filter_rules": [list(rule) for rule in ACTIVE_FILTER],
    }


def _post_load_corrections():
    """Validate loaded values against what the machine actually offers."""
    global SELECTED_CPU, MODEL_NAME, SELECTED_GPU, AVAILABLE_MODELS

    from scripts.utility import get_cpu_info
    cpu_labels = ["Auto-Select"] + [c["label"] for c in get_cpu_info()]
    if SELECTED_CPU not in cpu_labels:
        SELECTED_CPU = "Auto-Select"

    from scripts.inference import get_available_models
    AVAILABLE_MODELS = get_available_models()
    if MODEL_NAME not in AVAILABLE_MODELS:
        real_models = [m for m in AVAILABLE_MODELS if m != "Select_a_model..."]
        MODEL_NAME = real_models[0] if real_models else "Select_a_model..."

    if SELECTED_GPU is None or SELECTED_GPU == "Auto":
        try:
            from scripts.utility import get_available_gpus
            real_gpus = [g for g in get_available_gpus() if g != "CPU Only"]
            if len(real_gpus) == 1:
                SELECTED_GPU = real_gpus[0]
                print(f"[CONFIG] Auto-selected sole GPU: {SELECTED_GPU}")
            elif len(real_gpus) > 1:
                SELECTED_GPU = real_gpus[1]
                print(f"[CONFIG] Auto-selected secondary GPU: {SELECTED_GPU}")
            else:
                SELECTED_GPU = "Auto-Select"
        except Exception as e:
            print(f"[CONFIG] GPU auto-selection failed: {e}")
            SELECTED_GPU = "Auto-Select"


def load_config():
    """Load both settings files. configuration.json is required, preferences.json
    falls back to defaults so a missing or hand-deleted file is not fatal."""
    _apply_configuration(_read_settings(CONFIGURATION_PATH, "configuration", required=True))
    _apply_preferences(_read_settings(PREFERENCES_PATH, "preferences", required=False))

    print(f"[CONFIG] LOADING_MODE read as: {LOADING_MODE}")
    _post_load_corrections()
    print(f"[CONFIG] Loaded -> Model: {MODEL_NAME} | CPU: {SELECTED_CPU}")
    print(f"[CONFIG] Filter: {FILTER_MODE} ({len(ACTIVE_FILTER)} rules)")
    set_status("Configuration loaded", console=True)
    return "Configuration loaded."


def save_config():
    """Write the Configuration page settings to data/configuration.json."""
    _write_settings(CONFIGURATION_PATH, "configuration", _configuration_values())
    set_status("Configuration saved")
    return "Configuration saved"


def save_preferences():
    """Write the Preferences page settings to data/preferences.json."""
    _write_settings(PREFERENCES_PATH, "preferences", _preferences_values())
    set_status("Preferences saved")
    return "Preferences saved"


def restore_configuration_defaults(keep_model_folder: bool = True):
    """Reset every Configuration page setting, then write the file.

    The model folder is kept by default. It is the one setting a user cannot
    retype from memory, and clearing it would leave the model dropdown empty
    with no hint as to where the GGUF files went.
    """
    global MODEL_FOLDER
    folder = MODEL_FOLDER
    _apply_configuration(dict(CONFIGURATION_DEFAULTS))
    if keep_model_folder:
        MODEL_FOLDER = folder
    _post_load_corrections()
    _write_settings(CONFIGURATION_PATH, "configuration", _configuration_values())
    msg = "Configuration restored to defaults"
    if keep_model_folder:
        msg += " (model folder kept)"
    set_status(msg)
    return msg


def restore_preferences_defaults():
    """Reset every Preferences page setting, filter rules included, then save."""
    _apply_preferences(dict(PREFERENCES_DEFAULTS))
    _write_settings(PREFERENCES_PATH, "preferences", _preferences_values())
    set_status("Preferences restored to defaults")
    return "Preferences restored to defaults"


# =============================================================================
# STATUS FUNCTION
# =============================================================================

def set_status(msg: str, console=False, priority=False):
    """Update both UI and/or terminal with priority support."""
    global _status_lock, _status_lock_message

    if _status_lock and not priority:
        if console or len(msg.split()) > 3:
            print(f"[Background] {msg}")
        return

    if priority and ("Load" in msg or "loading" in msg.lower()):
        _status_lock = "model_loading"
        _status_lock_message = msg

    if _status_lock == "model_loading" and ("ready" in msg.lower() or "error" in msg.lower()):
        _status_lock = None
        _status_lock_message = ""

    if global_status is not None:
        global_status.value = msg

    if console or len(msg.split()) > 3:
        print(msg)