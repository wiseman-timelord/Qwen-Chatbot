# Script: installer.py - Installation script for Qwen-Chatbot
# Qwen-Chatbot: Windows 10 / Python 3.12 / Gradio 5.x / PyQt6 / Qwen GGUF models
# Note: Uses sentence-transformers for embeddings (offline)
# Note: Uses PyQt6 WebEngine for custom browser window

import os
import json
import subprocess
import sys
import contextlib
import time
import threading
import codecs
from pathlib import Path
import shutil
import atexit
import re

# Constants / Variables
_PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
APP_NAME = "Qwen-Chatbot"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
LLAMACPP_GIT_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMACPP_PYTHON_GIT_REPO = "https://github.com/abetlen/llama-cpp-python.git"
# llama-cpp-python version strategy:
#   LLAMACPP_PYTHON_PREBUILT_VERSION — last version with CPU prebuilt wheels
#     from eswarthammana (cpu-only, Win/Linux/Mac, cp38-cp313).
#     eswarthammana stopped publishing after v0.3.16 (Aug 2025).
LLAMACPP_PYTHON_PREBUILT_VERSION = "v0.3.16"
#   LLAMACPP_PYTHON_VERSION — resolved at install time to the latest GitHub
#     release tag. Falls back to LLAMACPP_PYTHON_VERSION_FALLBACK if unreachable.
LLAMACPP_PYTHON_VERSION = None
LLAMACPP_PYTHON_VERSION_FALLBACK = "v0.3.26" 
#   LLAMACPP_PYTHON_COMPILE_DISPLAY — display-only string for the backend menu.
#     Never used in pip commands; the compile path resolves the real version
#     from the GitHub API at install time.
LLAMACPP_PYTHON_COMPILE_DISPLAY = "v0.3.26"  
# Set during install_python_deps() once the wheel is confirmed installed.
# Written to constants.ini by update_ini_wheel_version() so the main program
# can display it in the About/Debug tab.
_INSTALLED_LLAMA_WHEEL_VERSION = None
LLAMACPP_TARGET_VERSION = "b9542"
WIN_COMPILE_TEMP = Path("C:/temp_build")
_INSTALL_PROCESSES = set()
_DID_COMPILATION = False
_PRE_EXISTING_PROCESSES = {}
_USER_BUILD_THREADS = None
PYTHON_VERSION = sys.version_info
WINDOWS_VERSION = None
_CPU_FEATURES = None
_CPU_DETECTED_EARLY = False
OS_VERSION = None
VS_GENERATOR = None

# Display/Browser variables
DX11_CAPABLE = None
DX_FEATURE_LEVEL = None
DX_FEATURE_NAME = None

# Maps/Lists
DIRECTORIES = [
    "data", "scripts", "models",
    "data/history", "data/temp", "data/vectors",
    "data/embedding_cache"
]

PROTECTED_DIRECTORIES = [
    "data/embedding_cache",
]

EMBEDDING_MODELS = {
    "1": {
        "name": "BAAI/bge-small-en-v1.5",
        "display": "Smaller/Faster Install - Bge-Small-English v1.5",
        "size_mb": 132
    },
    "2": {
        "name": "BAAI/bge-base-en-v1.5",
        "display": "Medium/Quality Install - Bge-Base-English v1.5",
        "size_mb": 425
    }
}

KOKORO_VOICE_PACKS = {
    "1": {
        "display": "American Male+Female Pack 2",
        "detail": "(2 male + 4 female)",
        "voices": "(Adam, Michael, Heart, Bella, Nova, Sky)",
        "voice_ids": ["am_adam", "am_michael", "af_heart", "af_bella", "af_nova", "af_sky"],
        "default_voice_id": "af_heart",
        "default_voice_name": "Heart — American Female",
        "lang_code": "a",
    },
    "2": {
        "display": "British Male+Female Pack 2",
        "detail": "(2 male + 2 female)",
        "voices": "(George, Lewis, Emma, Alice)",
        "voice_ids": ["bm_george", "bm_lewis", "bf_emma", "bf_alice"],
        "default_voice_id": "bm_george",
        "default_voice_name": "George — British Male",
        "lang_code": "b",
    },
}

# Functions
def print_status(message: str, success: bool = True) -> None:
    status = "[✓]" if success else "[✗]"
    print(f"{status} {message}")
    time.sleep(1 if success else 3)

def short_path(path, max_len=44):
    """Truncate path for display - installer standalone version"""
    path = str(path)
    if len(path) <= max_len:
        return path
    return "..." + path[-max_len:]

def detect_cpu_features() -> dict:
    """Detect CPU SIMD features accurately."""
    global _CPU_FEATURES, _CPU_DETECTED_EARLY

    if _CPU_FEATURES is not None:
        return _CPU_FEATURES

    features = {
        "AVX": False, "AVX2": False, "AVX512": False, "FMA": False,
        "F16C": False, "SSE3": False, "SSSE3": False, "SSE4_1": False, "SSE4_2": False
    }

    success = False

    try:
        import ctypes
        _ipfp = ctypes.windll.kernel32.IsProcessorFeaturePresent
        _ipfp.restype = ctypes.c_bool
        _ipfp.argtypes = [ctypes.c_uint]
        features["SSE3"]   = bool(_ipfp(13))
        features["SSSE3"]  = bool(_ipfp(36))
        features["SSE4_1"] = bool(_ipfp(37))
        features["SSE4_2"] = bool(_ipfp(38))
        features["AVX"]    = bool(_ipfp(39))
        features["AVX2"]   = bool(_ipfp(40))
        features["AVX512"] = bool(_ipfp(41))
        success = True
    except Exception:
        features["SSE3"] = True
        success = True

    if success:
        try:
            import cpuinfo as _cpuinfo
            _info  = _cpuinfo.get_cpu_info()
            _flags = [f.lower() for f in _info.get('flags', [])]
            features["FMA"]    = 'fma'  in _flags
            features["F16C"]   = 'f16c' in _flags
            features["AVX"]    = features["AVX"]    or ('avx'    in _flags)
            features["AVX2"]   = features["AVX2"]   or ('avx2'   in _flags)
            features["AVX512"] = features["AVX512"] or any('avx512' in f for f in _flags)
        except ImportError:
            pass

    if success:
        _CPU_FEATURES = features
        _CPU_DETECTED_EARLY = True

    return features


def detect_browser_acceleration() -> tuple:
    """
    Silent GPU/DX detection with caching.
    No printing — safe to call multiple times.
    """
    global DX11_CAPABLE, DX_FEATURE_LEVEL, DX_FEATURE_NAME

    if DX11_CAPABLE is not None:
        return (DX11_CAPABLE, DX_FEATURE_LEVEL)

    try:
        import ctypes

        d3d11 = ctypes.windll.LoadLibrary("d3d11.dll")
        feature_levels = (ctypes.c_uint * 4)(0xb100, 0xb000, 0xa100, 0xa000)
        device  = ctypes.c_void_p()
        fl_out  = ctypes.c_uint()
        ctx     = ctypes.c_void_p()

        hr = d3d11.D3D11CreateDevice(
            None, 1, None, 0, feature_levels, 4, 7,
            ctypes.byref(device), ctypes.byref(fl_out), ctypes.byref(ctx),
        )

        DX_FEATURE_LEVEL = fl_out.value
        DX_FEATURE_NAME  = {
            0xb100: "11.1", 0xb000: "11.0",
            0xa100: "10.1", 0xa000: "10.0"
        }.get(fl_out.value, f"0x{fl_out.value:04x}")
        DX11_CAPABLE = (hr == 0 and fl_out.value >= 0xb000)
        return (DX11_CAPABLE, DX_FEATURE_LEVEL)

    except:
        DX11_CAPABLE    = False
        DX_FEATURE_LEVEL = 0
        DX_FEATURE_NAME  = "Unknown"
        return (False, 0)


def run_initial_detection():
    run_detections_once()
    if _DETECTED_DX_LEVEL == 0:
        print_status("GPU acceleration: Not available (no DirectX 11)", False)
    else:
        print_status(f"GPU acceleration: DirectX {_DETECTED_DX_NAME}")
    print()


TEMP_DIR = WIN_COMPILE_TEMP

# Backend definitions
BACKEND_OPTIONS = {
    "Download CPU Binary / Default CPU Wheel": {
        "url": None, "dest": None, "cli_path": None,
        "needs_python_bindings": True, "compile_binary": False,
        "compile_wheel": False, "vulkan_required": False, "build_flags": {}
    },
    "Download Vulkan Binary / Default CPU Wheel": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-vulkan-x64.zip",
        "dest": "data/llama-vulkan-bin",
        "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
        "needs_python_bindings": True, "compile_binary": False,
        "compile_wheel": False, "vulkan_required": False, "build_flags": {}
    },
    "Compile CPU Binaries / Compile CPU Wheel": {
        "url": None, "dest": "data/llama-cpu-bin",
        "cli_path": "data/llama-cpu-bin/llama-cli.exe",
        "needs_python_bindings": True, "compile_binary": True,
        "compile_wheel": True, "vulkan_required": False, "build_flags": {}
    },
    "Compile Vulkan Binaries / Compile Vulkan Wheel": {
        "url": None, "dest": "data/llama-vulkan-bin",
        "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
        "needs_python_bindings": True, "compile_binary": True,
        "compile_wheel": True, "vulkan_required": True,
        "build_flags": {"GGML_VULKAN": "1"}
    }
}

# =============================================================================
# v2 BASE REQUIREMENTS
# =============================================================================
BASE_REQ = [
    "numpy>=2.0",
    "requests>=2.32.0",
    "pyperclip>=1.8.2",
    "spacy>=3.8.0",
    "psutil>=6.0.0",
    "ddgs>=9.10.0",
    "langchain>=0.3.18",            # install base first so langgraph/websockets resolve together
    "langchain-community>=0.3.18",
    "langchain-text-splitters>=0.3.0",
    "faiss-cpu>=1.9.0",
    "pygments>=2.17.0",
    "lxml>=5.2.0,<5.5.0",        # newspaper4k requires lxml<5.5; pin upfront to avoid downgrade
    "beautifulsoup4>=4.12.0",
    "aiohttp>=3.10.0",
    "newspaper4k>=0.9.4.1",      # installs lxml_html_clean as a dependency
    "lxml_html_clean>=0.3.0",    # explicit pin AFTER newspaper4k to avoid lxml 6.x pull
    "soundfile>=0.12.1",
    "kokoro>=0.9.4",
    "pywin32>=306",
    "tk==0.1.0",
    "pythonnet==3.0.5",
]

def clear_screen():
    os.system('cls')

def backend_requires_compilation(backend: str) -> bool:
    """Check if the selected backend requires compilation"""
    info = BACKEND_OPTIONS.get(backend, {})
    return info.get("compile_binary", False) or info.get("compile_wheel", False)


def get_installed_llama_info():
    """Return {'version': str, 'vulkan': bool} if llama_cpp imports in the venv, else None.

    Vulkan builds ship a ggml-vulkan DLL inside llama_cpp/lib, which
    distinguishes them from CPU-only wheels of the same version."""
    python_exe = VENV_DIR / "Scripts" / "python.exe"
    if not python_exe.exists():
        return None
    probe = (
        "import llama_cpp, pathlib; "
        "lib = pathlib.Path(llama_cpp.__file__).parent / 'lib'; "
        "vk = lib.exists() and any('vulkan' in p.name.lower() for p in lib.iterdir()); "
        "print(llama_cpp.__version__); print('vulkan' if vk else 'cpu')"
    )
    try:
        result = subprocess.run(
            [str(python_exe), "-c", probe],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if len(lines) >= 2:
                return {"version": lines[0].strip(), "vulkan": lines[1].strip() == "vulkan"}
    except Exception:
        pass
    return None


def detect_windows_version() -> str:
    """Detect Windows version and cache in global."""
    global WINDOWS_VERSION
    if WINDOWS_VERSION is not None:
        return WINDOWS_VERSION

    try:
        import platform
        version = platform.version()
        build = int(version.split('.')[-1])

        if build >= 22000:
            WINDOWS_VERSION = "11"
        elif build >= 10240:
            WINDOWS_VERSION = "10"
        else:
            WINDOWS_VERSION = "unsupported"
        return WINDOWS_VERSION
    except:
        WINDOWS_VERSION = "unknown"
        return "unknown"


def get_dynamic_requirements() -> list:
    requirements = BASE_REQ.copy()
    requirements.append("gradio>=5.0.0,<6.0.0")
    return requirements


def get_torch_version_for_python() -> str:
    return "torch>=2.5.0"

def check_version_compatibility():
    """Check Python and OS compatibility."""
    global WINDOWS_VERSION, PYTHON_VERSION

    if sys.version_info < (3, 11):
        print_status("Python 3.12 required for Qwen-Chatbot", False)
        return False

    if sys.version_info >= (3, 13):
        print_status(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported. "
            "Kokoro TTS requires Python <3.13. Please use Python 3.12.", False
        )
        return False

    PYTHON_VERSION = sys.version_info

    win_ver = detect_windows_version()
    if win_ver in ("unsupported", "7", "8", "8.1"):
        print_status(
            f"Windows {win_ver} is not supported. Requires Windows 10 or 11.", False
        )
        return False
    return True


def is_kokoro_compatible() -> bool:
    """Check if current OS/Python supports Kokoro TTS."""
    if sys.version_info >= (3, 13):
        return False
    return WINDOWS_VERSION in ["10", "11"]


# =============================================================================
# INSTALLATION HELPERS
# =============================================================================

def snapshot_pre_existing_processes() -> None:
    global _PRE_EXISTING_PROCESSES
    try:
        import psutil
    except ImportError:
        return

    build_process_names = [
        "conhost.exe", "MSBuild.exe", "VBCSCompiler.exe", "node.exe",
        "cmake.exe", "cl.exe", "link.exe", "lib.exe", "cvtres.exe",
        "mt.exe", "rc.exe", "mspdbsrv.exe", "vctip.exe", "tracker.exe",
        "git.exe", "python.exe", "pip.exe",
    ]
    _PRE_EXISTING_PROCESSES = {}
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] in build_process_names:
                _PRE_EXISTING_PROCESSES[proc.info['pid']] = proc.info['name']
    except Exception:
        pass


def track_process(pid: int) -> None:
    _INSTALL_PROCESSES.add(pid)


def cleanup_build_processes() -> None:
    try:
        import psutil
    except ImportError:
        return

    build_process_names = [
        "MSBuild.exe", "VBCSCompiler.exe", "cmake.exe", "cl.exe",
        "link.exe", "lib.exe", "cvtres.exe", "mt.exe", "rc.exe",
        "mspdbsrv.exe", "vctip.exe", "tracker.exe",
    ]

    for pid in list(_INSTALL_PROCESSES):
        try:
            proc = psutil.Process(pid)
            if proc.is_running():
                proc.terminate()
        except Exception:
            pass

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] in build_process_names:
                if proc.info['pid'] not in _PRE_EXISTING_PROCESSES:
                    proc.terminate()
        except Exception:
            pass


def _force_rmtree(path: Path) -> None:
    """Delete a directory tree, forcibly removing read-only files (Windows git repos)."""
    import stat

    def _on_error(func, fpath, exc_info):
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception:
            pass

    if path.exists():
        shutil.rmtree(str(path), onerror=_on_error)


def get_optimal_build_threads() -> int:
    """Return 85% of logical CPU cores for parallel builds."""
    if _USER_BUILD_THREADS is not None:
        return _USER_BUILD_THREADS
    import multiprocessing
    try:
        total = multiprocessing.cpu_count()
    except:
        total = 4
    return max(1, int(total * 0.85))


# =============================================================================
# MODULE-LEVEL DETECTION CACHE
# =============================================================================
_DETECTED_CPU_FEATURES: dict  = {}
_DETECTED_BUILD_TOOLS:  dict  = {}
_DETECTED_VULKAN:       bool  = False
_DETECTED_DX_CAPABLE:   bool  = False
_DETECTED_DX_LEVEL:     int   = 0
_DETECTED_DX_NAME:      str   = "Unknown"
_DETECTIONS_RUN:        bool  = False


def _find_cmake_in_vs_installations() -> str | None:
    """
    Search for cmake.exe inside Visual Studio / Build Tools installations
    (2019 and 2022, all editions) using vswhere, then by walking known paths.
    Returns the directory containing cmake.exe, or None if not found.
    """
    prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    prog_files     = os.environ.get("ProgramFiles",       r"C:\Program Files")

    # --- Strategy 1: ask vswhere for every install path (all products/versions) ---
    vswhere_exe = Path(prog_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    install_roots: list[str] = []

    if vswhere_exe.exists():
        try:
            result = subprocess.run(
                [
                    str(vswhere_exe),
                    "-all",               # all installed products
                    "-prerelease",        # include pre-release
                    "-property", "installationPath",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                install_roots = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        except Exception:
            pass

    # --- Strategy 2: hard-coded default roots for VS 2019 & 2022, all editions ---
    #     (catches standalone Build Tools whose vswhere entry may be missing)
    for base in (prog_files_x86, prog_files):
        for year in ("2022", "2019"):
            for edition in (
                "BuildTools",
                "Enterprise", "Professional", "Community", "Preview",
            ):
                candidate = os.path.join(base, "Microsoft Visual Studio", year, edition)
                if os.path.isdir(candidate) and candidate not in install_roots:
                    install_roots.append(candidate)

    # --- Walk each install root looking for cmake.exe under the CMake component ---
    for root in install_roots:
        cmake_bin = os.path.join(root, "Common7", "IDE", "CommonExtensions",
                                 "Microsoft", "CMake", "CMake", "bin")
        cmake_exe = os.path.join(cmake_bin, "cmake.exe")
        if os.path.isfile(cmake_exe):
            return cmake_bin   # found – return the bin directory

    return None


def detect_build_tools_available() -> dict:
    """Detect availability of Git, CMake, MSVC, MSBuild."""
    tools = {"Git": False, "CMake": False, "MSVC": False, "MSBuild": False}

    # Git ------------------------------------------------------------------
    if shutil.which("git"):
        tools["Git"] = True

    # CMake ----------------------------------------------------------------
    # Priority 1: already on PATH
    if shutil.which("cmake"):
        tools["CMake"] = True
    else:
        # Priority 2: bundled inside VS / Build Tools 2019-2022
        cmake_bin = _find_cmake_in_vs_installations()
        if cmake_bin:
            tools["CMake"] = True
            # Prepend to PATH so cmake is usable by any subsequent subprocess
            os.environ["PATH"] = cmake_bin + os.pathsep + os.environ.get("PATH", "")

    # MSVC / MSBuild ---------------------------------------------------------
    try:
        vswhere = (Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
                   / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
        if vswhere.exists():
            result = subprocess.run(
                [str(vswhere), "-latest", "-property", "installationPath"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                tools["MSVC"] = True
                msbuild_path = (Path(result.stdout.strip())
                               / "MSBuild" / "Current" / "Bin" / "MSBuild.exe")
                if msbuild_path.exists():
                    tools["MSBuild"] = True
    except Exception:
        pass

    return tools


def run_detections_once() -> None:
    """Run all hardware/tool detections exactly once and cache results."""
    global _DETECTED_CPU_FEATURES, _DETECTED_BUILD_TOOLS, _DETECTED_VULKAN
    global _DETECTED_DX_CAPABLE, _DETECTED_DX_LEVEL, _DETECTED_DX_NAME, _DETECTIONS_RUN

    if _DETECTIONS_RUN:
        return

    try:
        _DETECTED_CPU_FEATURES = detect_cpu_features()
    except Exception:
        _DETECTED_CPU_FEATURES = {}

    try:
        _DETECTED_BUILD_TOOLS = detect_build_tools_available()
    except Exception:
        _DETECTED_BUILD_TOOLS = {"Git": False, "CMake": False, "MSVC": False, "MSBuild": False}

    try:
        _DETECTED_VULKAN = is_vulkan_installed()
    except Exception:
        _DETECTED_VULKAN = False

    try:
        _DETECTED_DX_CAPABLE, _DETECTED_DX_LEVEL = detect_browser_acceleration()
        _DETECTED_DX_NAME = DX_FEATURE_NAME or "Unknown"
    except Exception:
        _DETECTED_DX_CAPABLE  = False
        _DETECTED_DX_LEVEL    = 0
        _DETECTED_DX_NAME     = "Unknown"

    _DETECTIONS_RUN = True
    print(f"[DETECT] CPU: {', '.join(k for k,v in _DETECTED_CPU_FEATURES.items() if v) or 'baseline'}")
    print(f"[DETECT] Vulkan: {'YES' if _DETECTED_VULKAN else 'NO'} | DX: {_DETECTED_DX_NAME}")
    print(f"[DETECT] Build tools: {', '.join(k for k,v in _DETECTED_BUILD_TOOLS.items() if v) or 'none'}")

def print_header(section: str = "Initialization") -> None:
    clear_screen()
    width = shutil.get_terminal_size().columns - 1
    print("=" * width)
    print(f"    Qwen-Chatbot — {section}")
    print("=" * width)
    print()


def create_files_and_directories(backend: str) -> None:
    for directory in DIRECTORIES:
        dir_path = BASE_DIR / directory
        if str(dir_path) in [str(BASE_DIR / p) for p in PROTECTED_DIRECTORIES]:
            if dir_path.exists():
                print_status(f"Protected directory preserved: {directory}")
                continue
        dir_path.mkdir(parents=True, exist_ok=True)

    # TEMP_DIR cannot go in DIRECTORIES above: those are all relative to
    # BASE_DIR, and TEMP_DIR is C:/temp_build (WIN_COMPILE_TEMP, kept short
    # so build paths do not blow the path length limit).
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    print_status(f"Directories created/verified (temp: {TEMP_DIR})")


# =============================================================================
# EMBEDDING BACKEND INSTALLATION
# =============================================================================
# Order matters. kokoro (in BASE_REQ) and sentence-transformers both depend on
# torch. If torch is absent when they install, pip pulls the default PyPI
# Windows wheel (no +cpu tag). That fights the intentional CPU build from
# download.pytorch.org (symptom: "No module named 'torch'" at verification).
#
# Fix:
#   1. install_torch_cpu() runs FIRST, before BASE_REQ.
#   2. After transformers / sentence-transformers, re-assert the CPU torch.
#   3. Verify import with the venv python.exe (full path — no activation needed).

def install_torch_cpu() -> bool:
    """Install the official CPU torch wheel from pytorch.org before torch deps."""
    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
    torch_req = "torch>=2.5.0"
    print_status(f"Installing PyTorch (CPU) — {torch_req}...")
    if not pip_install_with_retry(
        pip_exe, torch_req,
        ["--index-url", "https://download.pytorch.org/whl/cpu",
         "--upgrade-strategy", "only-if-needed"],
        max_retries=10, initial_delay=5.0,
    ):
        print_status("PyTorch installation failed", False)
        return False
    print_status("PyTorch (CPU) installed")
    subprocess.run(
        [pip_exe, "install", "setuptools>=80.0", "--upgrade", "--quiet"],
        capture_output=True, timeout=120,
    )
    print_status("setuptools restored after torch install")
    return True


def _reassert_torch_cpu() -> bool:
    """Force the CPU torch build after dependency packages may have overwritten it."""
    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
    torch_req = "torch>=2.5.0"
    print_status("Re-asserting PyTorch (CPU) after dependency packages...")
    if not pip_install_with_retry(
        pip_exe, torch_req,
        ["--index-url", "https://download.pytorch.org/whl/cpu",
         "--force-reinstall", "--no-deps"],
        max_retries=5, initial_delay=3.0,
    ):
        print_status("Failed to re-assert PyTorch (CPU)", False)
        return False
    print_status("PyTorch (CPU) re-asserted")
    return True


def install_embedding_backend() -> bool:
    """Install transformers + sentence-transformers, lock CPU torch, verify imports."""
    python_exe = str(VENV_DIR / "Scripts" / "python.exe")
    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
    transformers_version = "transformers>=4.44.0"
    sentence_transformers_version = "sentence-transformers>=3.3.0"

    print_status(f"Installing {transformers_version}...")
    if not pip_install_with_retry(
        pip_exe, transformers_version,
        ["--upgrade-strategy", "only-if-needed"],
        max_retries=10, initial_delay=5.0,
    ):
        print_status("transformers installation failed", False)
        return False
    print_status("transformers installed")

    print_status(f"Installing {sentence_transformers_version}...")
    if not pip_install_with_retry(
        pip_exe, sentence_transformers_version,
        ["--upgrade-strategy", "only-if-needed"],
        max_retries=10, initial_delay=5.0,
    ):
        print_status("sentence-transformers installation failed", False)
        return False
    print_status("sentence-transformers installed")

    if not _reassert_torch_cpu():
        return False

    verify_script = (
        "import sys, os\n"
        "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
        "try:\n"
        "    import torch\n"
        "    torch.set_grad_enabled(False)\n"
        "    print(f'torch: {torch.__version__}')\n"
        "    from sentence_transformers import SentenceTransformer\n"
        "    print('sentence_transformers: OK')\n"
        "    print('SUCCESS')\n"
        "except Exception as e:\n"
        "    print(f'FAILED: {e}')\n"
        "    print(f'python: {sys.executable}')\n"
        "    try:\n"
        "        import importlib.util\n"
        "        print(f'torch spec: {importlib.util.find_spec(\"torch\")}')\n"
        "    except Exception as e2:\n"
        "        print(f'torch spec error: {e2}')\n"
        "    sys.exit(1)\n"
    )
    verify_path = TEMP_DIR / "verify_embedding.py"
    try:
        verify_path.parent.mkdir(parents=True, exist_ok=True)
        with open(verify_path, "w", encoding="utf-8") as f:
            f.write(verify_script)
        verify_result = subprocess.run(
            [python_exe, str(verify_path)],
            capture_output=True, text=True, timeout=120,
        )
        verify_path.unlink(missing_ok=True)
        out = (verify_result.stdout or "") + (verify_result.stderr or "")
        if verify_result.returncode == 0 and "SUCCESS" in out:
            print_status("Embedding backend verified")
            return True
        print_status("Embedding backend verification failed", False)
        print(f"Output: {out.strip()}")
        return False
    except Exception as e:
        print_status(f"Verification error: {e}", False)
        return False


# =============================================================================
# QT WEBENGINE INSTALLATION
# =============================================================================

def install_qt_webengine() -> bool:
    """Install PyQt6 + PyQt6-WebEngine for the custom browser window."""
    print_status("Installing Qt6 WebEngine for custom browser...")
    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")

    try:
        if not pip_install_with_retry(pip_exe, "PyQt6>=6.6.0", max_retries=3, initial_delay=5.0):
            print_status("PyQt6 installation failed - will use system browser", False)
            return False

        if not pip_install_with_retry(pip_exe, "PyQt6-WebEngine>=6.6.0", max_retries=3, initial_delay=5.0):
            print_status("PyQt6-WebEngine installation failed - will use system browser", False)
            return False

        print_status("Qt6 WebEngine installed successfully")
        return True

    except Exception as e:
        print_status(f"Qt WebEngine error: {e} - will use system browser", False)
        return False


# =============================================================================
# PYTHON DEPENDENCY INSTALLATION
# =============================================================================

def install_python_deps(backend: str, skip_if_present: bool = False) -> bool:
    """Install Python dependencies.

    skip_if_present: on Check/Install runs, skip the llama-cpp-python
    download/compile when a matching wheel is already in the venv."""
    global _INSTALLED_LLAMA_WHEEL_VERSION
    print_status("Installing Python dependencies...")
    try:
        pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")

        all_requirements = get_dynamic_requirements()

        # CPU torch MUST be present before BASE_REQ (kokoro depends on torch).
        if not install_torch_cpu():
            return False

        print_status(f"Installing Python packages...")
        total = len(all_requirements)
        for i, req in enumerate(all_requirements, 1):
            pkg_name = req.split('>=')[0].split('==')[0].split('[')[0]
            print(f"  [{i}/{total}] Installing {pkg_name}...  ", end='', flush=True)

            if pip_install_with_retry(pip_exe, req, max_retries=10, initial_delay=5.0):
                print(f" OK")
            else:
                print(f" FAILED")
                print_status(f"Failed to install {pkg_name} after 10 retries", False)
                return False

        print_status("Base packages installed")

        if not install_embedding_backend():
            return False

        install_qt_webengine()

        info = BACKEND_OPTIONS[backend]

        existing = get_installed_llama_info() if skip_if_present else None

        if not info.get("compile_wheel"):
            wheel_version = LLAMACPP_PYTHON_PREBUILT_VERSION.lstrip("v")

            if existing and existing["version"] == wheel_version and not existing["vulkan"]:
                print_status(f"llama-cpp-python {wheel_version} (CPU wheel) already installed - skipping download")
                _INSTALLED_LLAMA_WHEEL_VERSION = f"v{wheel_version}"
                print_status("Python dependencies installed successfully")
                return True

            sources = _get_prebuilt_wheel_urls()

            if not sources:
                print_status("No pre-built wheel sources available.", False)
                return False

            print_status(f"Installing llama-cpp-python {wheel_version} (CPU, trying {len(sources)} sources)...")
            installed = False

            for src in sources:
                label = src.get("label", src["value"])
                print(f"  Trying: {label}")

                if src["type"] == "url":
                    installed = pip_install_with_retry(pip_exe, src["value"], max_retries=2, initial_delay=3.0)
                elif src["type"] == "index":
                    installed = pip_install_with_retry(
                        pip_exe, src["value"],
                        extra_args=["--extra-index-url", src["extra_index"], "--prefer-binary"],
                        max_retries=3, initial_delay=5.0
                    )
                elif src["type"] == "pypi":
                    installed = pip_install_with_retry(
                        pip_exe, src["value"],
                        extra_args=["--prefer-binary"],
                        max_retries=3, initial_delay=5.0
                    )

                if installed:
                    print_status(f"llama-cpp-python {wheel_version} installed via {label}")
                    _INSTALLED_LLAMA_WHEEL_VERSION = f"v{wheel_version}"
                    break
                else:
                    print(f"  Source unavailable: {label}")

            if not installed:
                print_status(f"llama-cpp-python {wheel_version} could not be installed from any prebuilt source.", False)
                return False
        else:
            build_flags = info.get("build_flags", {})
            needs_vulkan = bool(build_flags.get("GGML_VULKAN"))

            if existing and existing["vulkan"] == needs_vulkan:
                build_kind = "Vulkan" if needs_vulkan else "CPU"
                print_status(f"llama-cpp-python {existing['version']} ({build_kind} build) already installed - skipping compile")
                _INSTALLED_LLAMA_WHEEL_VERSION = f"v{existing['version']}"
                print_status("Python dependencies installed successfully")
                return True

            if build_flags.get("GGML_VULKAN"):
                print_status("Vulkan wheel build - checking Vulkan SDK...")
                if not check_vulkan_sdk_installed():
                    print_status("Error: Vulkan SDK not found", False)
                    return False

            if not check_vcredist_windows():
                print_status("Warning: Visual C++ Redistributable (x64) not detected", False)
                time.sleep(3)

            if not build_llama_cpp_python_with_flags(build_flags):
                return False
            _INSTALLED_LLAMA_WHEEL_VERSION = LLAMACPP_PYTHON_VERSION

        print_status("Python dependencies installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print_status(f"Install failed: {e}", False)
        return False


def install_optional_file_support() -> bool:
    """Install optional file format libraries"""
    print_status("Installing optional file format support...")
    optional_packages = [
         "PyPDF2>=3.0.0",
         "python-docx>=1.1.0",
         "openpyxl>=3.1.0",
         "python-pptx>=1.0.0",
    ]

    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")

    for package in optional_packages:
        try:
            subprocess.run([pip_exe, "install", package], check=True, capture_output=True)
            print_status(f"  Installed {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print_status(f"  Optional package {package.split('>=')[0]} failed", False)
            return False  # <--- CHANGED: Return False instead of continuing

    return True


# =============================================================================
# SYSTEM INI CREATION
# =============================================================================

def create_system_ini(os_version, python_version,
                      backend_type, embedding_model,
                      windows_version=None, vulkan_available=False,
                      llama_cli_path=None, llama_bin_path=None,
                      tts_engine="kokoro",
                      tts_pack=1, tts_default_voice_id=None,
                      tts_default_voice_name=None, tts_enabled_voices=None,
                      browser_acceleration=True,
                      dx_feature_level=0,
                      install_method=None,
                      backend_key=None,
                      llama_wheel_vulkan=None):
    """Write data/constants.ini with system, TTS, and last-install metadata."""
    from datetime import datetime as _dt
    system_ini_path = BASE_DIR / "data" / "constants.ini"
    try:
        with open(system_ini_path, "w", encoding='utf-8') as f:
            f.write("[system]\n")
            f.write(f"platform = windows\n")
            f.write(f"os_version = {os_version}\n")
            f.write(f"python_version = {python_version}\n")
            f.write(f"backend_type = {backend_type}\n")
            f.write(f"embedding_model = {embedding_model}\n")
            f.write(f"embedding_backend = sentence_transformers\n")
            f.write(f"vulkan_available = {str(vulkan_available).lower()}\n")
            f.write(f"browser_acceleration = {str(browser_acceleration).lower()}\n")
            f.write(f"qt_version = 6\n")
            f.write(f"dx_feature_level = {dx_feature_level}\n")
            f.write(f"gradio_version = 5.x\n")
            if llama_cli_path:
                f.write(f"llama_cli_path = {llama_cli_path}\n")
            if llama_bin_path:
                f.write(f"llama_bin_path = {llama_bin_path}\n")
            if windows_version:
                f.write(f"windows_version = {windows_version}\n")
            if _INSTALLED_LLAMA_WHEEL_VERSION:
                f.write(f"llama_wheel_version = {_INSTALLED_LLAMA_WHEEL_VERSION}\n")

            f.write("\n[tts]\n")
            f.write(f"tts_type = {tts_engine}\n")
            f.write(f"tts_pack = {tts_pack}\n")
            f.write(f"tts_default_voice_id = {tts_default_voice_id or 'af_heart'}\n")
            f.write(f"tts_default_voice_name = {tts_default_voice_name or 'Heart — American Female'}\n")
            if tts_enabled_voices:
                f.write(f"tts_enabled_voices = {','.join(tts_enabled_voices)}\n")
            else:
                all_ids = []
                for pack in KOKORO_VOICE_PACKS.values():
                    all_ids.extend(pack["voice_ids"])
                f.write(f"tts_enabled_voices = {','.join(all_ids)}\n")

            f.write("\n[install]\n")
            f.write(f"last_success = {_dt.now().isoformat(timespec='seconds')}\n")
            f.write(f"last_method = {install_method or 'unknown'}\n")
            f.write(f"backend_key = {backend_key or ''}\n")
            f.write(f"backend_type = {backend_type}\n")
            if _INSTALLED_LLAMA_WHEEL_VERSION:
                f.write(f"llama_wheel_version = {_INSTALLED_LLAMA_WHEEL_VERSION}\n")
            if llama_wheel_vulkan is not None:
                f.write(f"llama_wheel_vulkan = {str(bool(llama_wheel_vulkan)).lower()}\n")

        print_status("System information file created")
        return True
    except Exception as e:
        print_status(f"Failed to create constants.ini: {str(e)}", False)
        return False



def update_ini_wheel_version(version: str) -> bool:
    """Patch constants.ini to record the llama-cpp-python wheel version."""
    import configparser as _cp
    ini_path = BASE_DIR / "data" / "constants.ini"
    if not ini_path.exists():
        print_status("constants.ini not found — cannot record wheel version", False)
        return False
    try:
        cfg_ini = _cp.ConfigParser()
        cfg_ini.read(ini_path, encoding='utf-8')
        if 'system' not in cfg_ini:
            print_status("constants.ini missing [system] — cannot record wheel version", False)
            return False
        cfg_ini['system']['llama_wheel_version'] = version
        with open(ini_path, 'w', encoding='utf-8') as f:
            cfg_ini.write(f)
        print_status(f"Recorded llama-cpp-python wheel version: {version}")
        return True
    except Exception as e:
        print_status(f"Could not update wheel version in constants.ini: {e}", False)
        return False


def create_venv() -> bool:
    try:
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
            print_status("Removed existing virtual environment")

        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

        print_status("Created new virtual environment")

        python_exe = VENV_DIR / "Scripts" / "python.exe"
        pip_exe    = VENV_DIR / "Scripts" / "pip.exe"

        if not python_exe.exists():
            raise FileNotFoundError(f"Python executable not found at {python_exe}")

        # Use 'python -m pip' (not the pip shim) — the correct, reliable upgrade path.
        # Stream output so failures are visible rather than silently swallowed.
        pip_upgrade = subprocess.run(
            [str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
            capture_output=True, text=True, timeout=120
        )
        if pip_upgrade.returncode != 0:
            print(f"  pip upgrade warning: {pip_upgrade.stderr.strip()[:200]}")
        print_status("Upgraded pip to latest version")
        print_status("Verified virtual environment setup")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to create venv: {e}", False)
        return False


def ensure_venv() -> bool:
    if VENV_DIR.exists():
        print_status("Existing virtual environment found - skipping recreation")
        return True
    return create_venv()


def simple_progress_bar(current: int, total: int, width: int = 25) -> str:
    if total == 0:
        return "[" + "=" * width + "] 100%"
    filled_width = int(width * current // total)
    bar = "=" * filled_width + "-" * (width - filled_width)
    percent = 100 * current // total

    def format_bytes(b):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if b < 1024.0:
                return f"{b:.1f}{unit}"
            b /= 1024.0
        return f"{b:.1f}TB"

    return f"[{bar}] {percent}% ({format_bytes(current)}/{format_bytes(total)})"

def check_vcredist_windows() -> bool:
    try:
        import winreg
        key_paths = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        ]
        for key_path in key_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path):
                    return True
            except FileNotFoundError:
                continue
        return False
    except:
        return False


def check_vulkan_sdk_installed() -> bool:
    vulkan_sdk = os.environ.get("VULKAN_SDK")
    if vulkan_sdk and Path(vulkan_sdk).is_dir():
        return True
    default_sdk = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "VulkanSDK"
    if default_sdk.exists():
        for child in default_sdk.iterdir():
            if child.is_dir() and (child / "Bin" / "vulkaninfoSDK.exe").exists():
                os.environ["VULKAN_SDK"] = str(child)
                return True
    return False


def is_vulkan_installed() -> bool:
    """Check if Vulkan runtime is installed on the system."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SOFTWARE\Khronos\Vulkan\Drivers")
        _, num_values, _ = winreg.QueryInfoKey(key)
        winreg.CloseKey(key)
        if num_values > 0:
            return True
    except Exception:
        pass

    if shutil.which("vulkaninfo"):
        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=8
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass

    vulkan_sdk_env = os.environ.get("VULKAN_SDK", "")
    if vulkan_sdk_env and Path(vulkan_sdk_env).is_dir():
        return True

    pf = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    for folder_name in ("VulkanRT", "VulkanSDK"):
        folder = pf / folder_name
        if folder.exists() and any(folder.iterdir()):
            return True

    sys32 = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"
    if (sys32 / "vulkan-1.dll").exists():
        return True

    return False


def pip_install_with_retry(pip_exe: str, package: str, extra_args: list = None,
                           max_retries: int = 10, initial_delay: float = 5.0,
                           force_reinstall: bool = False, no_deps: bool = False) -> bool:
    """Install a pip package with retry logic and exponential backoff."""
    INACTIVITY_TIMEOUT = 300
    _PROGRESS_KEYWORDS = ("downloading", "installing", "collected", "building",
                          "error", "warning", "failed", "%")
    _SUPPRESS_WARNINGS = ("pip's dependency resolver does not currently take into account",)

    if extra_args is None:
        extra_args = []

    pkg_name = package.split(">=")[0].split("==")[0].split("[")[0]
    delay = initial_delay

    install_flags = []
    if force_reinstall:
        install_flags.append("--force-reinstall")
    if no_deps:
        install_flags.append("--no-deps")

    for attempt in range(max_retries):
        cmd = [pip_exe, "install"] + install_flags + [package] + extra_args
        all_output: list[str] = []
        last_activity = [time.time()]
        reader_done  = [False]
        stall_reason: list[str] = [None]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )

            def _read_output():
                try:
                    for raw_line in proc.stdout:
                        line = raw_line.rstrip()
                        if not line:
                            continue
                        if any(kw in line.lower() for kw in _SUPPRESS_WARNINGS):
                            continue
                        last_activity[0] = time.time()
                        all_output.append(line)
                        if any(kw in line.lower() for kw in _PROGRESS_KEYWORDS):
                            print(f"    {line}", flush=True)
                finally:
                    reader_done[0] = True

            reader = threading.Thread(target=_read_output, daemon=True)
            reader.start()

            while not reader_done[0]:
                time.sleep(2)
                idle = time.time() - last_activity[0]
                if idle >= INACTIVITY_TIMEOUT:
                    stall_reason[0] = f"No output for {idle:.0f}s — connection stalled"
                    proc.kill()
                    break

            reader.join(timeout=5)
            proc.wait()

            combined = "\n".join(all_output).lower()

            if proc.returncode == 0 or "already satisfied" in combined:
                return True

            if stall_reason[0]:
                reason = stall_reason[0]
            else:
                error_lines = [l for l in all_output if "error" in l.lower()]
                reason = (f"pip error — {error_lines[-1][:120]}" if error_lines
                         else f"pip exited with code {proc.returncode}")

            if attempt < max_retries - 1:
                print(f"    Reason: {reason}")
                print(f"    Retry {attempt + 1}/{max_retries} for {pkg_name} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Unexpected error: {e}")
                print(f"    Retry {attempt + 1}/{max_retries} for {pkg_name} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)

    return False


# =============================================================================
# PREBUILT WHEEL URL HELPERS
# =============================================================================

def _get_prebuilt_wheel_urls() -> list:
    """Return ordered list of prebuilt wheel sources to try."""
    wheel_version = LLAMACPP_PYTHON_PREBUILT_VERSION.lstrip("v")
    py_tag = _PY_TAG
    sources = []

    filename = f"llama_cpp_python-{wheel_version}-{py_tag}-{py_tag}-win_amd64.whl"
    sources.append({
        "type": "url",
        "label": f"eswarthammana/llama-cpp-wheels {wheel_version}",
        "value": f"https://github.com/eswarthammana/llama-cpp-wheels/releases/download/{LLAMACPP_PYTHON_PREBUILT_VERSION}/{filename}"
    })

    sources.append({
        "type": "pypi",
        "label": f"PyPI llama-cpp-python {wheel_version}",
        "value": f"llama-cpp-python=={wheel_version}"
    })

    return sources


def get_latest_llamacpp_python_version() -> str:
    """Fetch the latest llama-cpp-python release tag from GitHub.
    Returns a tag like 'v0.3.26' — always a real version that pip accepts.
    Filters out pre-release/special tags like v0.3.26-hip-radeon."""
    try:
        import requests
        import re
        
        # Use /releases (not /releases/latest) to get all releases
        response = requests.get(
            "https://api.github.com/repos/abetlen/llama-cpp-python/releases",
            timeout=10
        )
        
        if response.status_code == 200:
            releases = response.json()
            
            # Pattern for standard version tags: v0.3.26 or 0.3.26
            # Excludes tags with suffixes like -hip-radeon, -cuda, -rc1, etc.
            version_pattern = re.compile(r'^v?\d+\.\d+\.\d+$')
            
            # First, try to find a non-prerelease standard version
            for release in releases:
                tag = release.get("tag_name", "")
                if release.get("prerelease"):
                    continue
                if version_pattern.match(tag):
                    print(f"[GITHUB] Latest llama-cpp-python release: {tag}")
                    return tag
            
            # If no non-prerelease standard version found, try all releases
            for release in releases:
                tag = release.get("tag_name", "")
                if version_pattern.match(tag):
                    print(f"[GITHUB] Latest llama-cpp-python release (pre-release): {tag}")
                    return tag
            
            # If still no valid version found, use fallback
            print(f"[GITHUB] No valid version found, using fallback: {LLAMACPP_PYTHON_VERSION_FALLBACK}")
            return LLAMACPP_PYTHON_VERSION_FALLBACK
        else:
            print(f"[GITHUB] Failed to fetch releases (HTTP {response.status_code}), using fallback")
            return LLAMACPP_PYTHON_VERSION_FALLBACK
            
    except Exception as e:
        print(f"[GITHUB] Failed to fetch latest version: {e}")
        return LLAMACPP_PYTHON_VERSION_FALLBACK


def build_llama_cpp_python_with_flags(build_flags: dict) -> bool:
    """Build llama-cpp-python from source with the given CMAKE flags.
    Version is resolved from the GitHub API (real PyPI version), never
    from the display-only LLAMACPP_PYTHON_COMPILE_DISPLAY constant."""
    global LLAMACPP_PYTHON_VERSION
    print_status("Compiling llama-cpp-python from source (this may take a while)...")

    # Ensure cmake is reachable — detection may have found it inside VS Build
    # Tools but not yet injected it into PATH (e.g. if build is called directly).
    if not shutil.which("cmake"):
        cmake_bin = _find_cmake_in_vs_installations()
        if cmake_bin:
            os.environ["PATH"] = cmake_bin + os.pathsep + os.environ.get("PATH", "")
            print_status(f"CMake located at: {cmake_bin}")

    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
    
    # Resolve version — always a real PyPI-compatible version string
    if LLAMACPP_PYTHON_VERSION is None:
        LLAMACPP_PYTHON_VERSION = get_latest_llamacpp_python_version()
    
    # Strip leading 'v' for pip; the tag is like 'v0.3.26' → '0.3.26'
    raw_version = LLAMACPP_PYTHON_VERSION.lstrip("v")
    pkg_spec = f"llama-cpp-python=={raw_version}"
    
    print_status(f"Building llama-cpp-python version {raw_version} from source...")
    
    env = os.environ.copy()
    cmake_args = []
    for key, value in build_flags.items():
        cmake_args.append(f"-D{key}={value}")
    
    if cmake_args:
        env["CMAKE_ARGS"] = " ".join(cmake_args)
        print_status(f"CMAKE_ARGS: {env['CMAKE_ARGS']}")
    
    env["FORCE_CMAKE"] = "1"
    
    try:
        proc = subprocess.Popen(
            [pip_exe, "install", pkg_spec, "--no-cache-dir", "--force-reinstall", "--verbose"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in proc.stdout:
            line = line.rstrip()
            if any(kw in line.lower() for kw in ("error", "failed", "building", "installing", "success")):
                print(f"    {line}", flush=True)
        
        proc.wait()
        
        if proc.returncode == 0:
            print_status("llama-cpp-python compiled and installed successfully")
            return True
        else:
            print_status("llama-cpp-python compilation failed", False)
            return False
            
    except Exception as e:
        print_status(f"Compilation error: {e}", False)
        return False


# =============================================================================
# NEW MENU FUNCTIONS
# =============================================================================

def _get_menu_choice(num_options: int, prompt: str = None) -> str:
    """Get and validate menu selection. Returns '1'-'N' or 'A' for abandon."""
    while True:
        try:
            raw = input(prompt or f"Selection; Menu Options = 1-{num_options}, Abandon = A: ").strip()
            if raw.upper() == 'A':
                return 'A'
            if raw.isdigit() and 1 <= int(raw) <= num_options:
                return raw
            print(f"  Invalid selection. Enter 1-{num_options} or A to abandon.")
        except (KeyboardInterrupt, EOFError):
            print()
            return 'A'


def show_main_menu() -> str:
    """Display the first installation menu with system detections."""
    run_detections_once()

    width = shutil.get_terminal_size().columns - 1

    print_header("Install Method")

    # ── System Detections ──────────────────────────────────────────────
    print("System Detections...")

    # CPU Features
    cpu_feats = [k for k, v in _DETECTED_CPU_FEATURES.items() if v]
    cpu_str = " | ".join(cpu_feats) if cpu_feats else "baseline"
    print(f"   CPU Features : {cpu_str}")

    # Build Tools
    build_ok = [k for k, v in _DETECTED_BUILD_TOOLS.items() if v]
    build_str = " | ".join(f"{k} OK" for k in build_ok) if build_ok else "none detected"
    print(f"   Build Tools  : {build_str}")

    # Platform
    win_ver = WINDOWS_VERSION or detect_windows_version() or "unknown"
    plat_str = f"Windows {win_ver}"
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"   Platform     : {plat_str} | Python {py_ver}")

    # GPU
    gpu_parts = []
    if _DETECTED_DX_NAME and _DETECTED_DX_NAME != "Unknown":
        gpu_parts.append(f"DX{_DETECTED_DX_NAME}")
    gpu_parts.append(f"Vulkan: {'YES' if _DETECTED_VULKAN else 'NO'}")
    print(f"   GPU          : {' | '.join(gpu_parts)}")

    print()
    print("-" * width)
    print()
    print("   1. Clean/Purge Install All")
    print()
    print("   2. Check/Install Python/Libraries")
    print()
    print("   3. Check/Recompile Binaries")
    print()
    print("   4. Refresh Configs/Inis")
    print()
    print()
    print("=" * width)

    return _get_menu_choice(4, "Selection; Menu Options = 1-4, Abandon Install = A: ")


def show_embedding_menu() -> str:
    """Display embedding model selection menu."""
    width = shutil.get_terminal_size().columns - 1

    print_header("Embeddings Size")
    print()
    print()
    print()
    print()
    print()
    print()
    print("   1) Smaller/Faster Install - Bge-Small-English v1.5")
    print()
    print("   2) Medium/Quality Install - Bge-Base-English v1.5")
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print("=" * width)

    return _get_menu_choice(2, "Selection; Menu Options = 1-2, Abandon Install = A: ")


def show_backend_menu() -> str:
    """Display backend selection menu with wheel version info."""
    width = shutil.get_terminal_size().columns - 1

    print_header("Llama.Cpp Backend")
    print()
    print()
    print()
    print()

    # Display-only versions — LLAMACPP_PYTHON_COMPILE_DISPLAY is for the menu
    # only and is NEVER passed to pip.  The compile path resolves the real
    # version from the GitHub API at build time.
    prebuilt_ver = LLAMACPP_PYTHON_PREBUILT_VERSION
    compile_ver = LLAMACPP_PYTHON_COMPILE_DISPLAY

    cmake_available = _DETECTED_BUILD_TOOLS.get("CMake", False)

    print(f"   1) Download CPU Binary / Default CPU Wheel (Wheel {prebuilt_ver})")
    print()
    print(f"   2) Download Vulkan Binary / Default CPU Wheel (Wheel {prebuilt_ver})")
    print()
    if cmake_available:
        print(f"   3) Compile CPU Binaries / Compile CPU Wheel (Wheel {compile_ver})")
        print()
        print(f"   4) Compile Vulkan Binaries / Compile Vulkan Wheel (Wheel {compile_ver})")
    else:
        print(f"   (Compile options 3 & 4 require CMake — not detected)")
        print()
        print(f"   Install CMake via Visual Studio Installer (C++ CMake tools) and")
        print(f"   RESTART this terminal, or download from https://cmake.org/download/")
    print()
    print()
    print()
    print()
    print()
    print("=" * width)

    if cmake_available:
        return _get_menu_choice(4, "Selection; Menu Options =1-4, Abandon=A: ")
    else:
        return _get_menu_choice(2, "Selection; Menu Options =1-2, Abandon=A: ")


def show_tts_menu() -> str:
    """Display TTS voice pack selection menu."""
    width = shutil.get_terminal_size().columns - 1

    clear_screen()
    print("=" * width)
    print(" Kokoro TTS — Voice Pack Selection")
    print("=" * width)
    print()
    print()
    print()
    print()
    print()
    print()
    
    for key in sorted(KOKORO_VOICE_PACKS.keys()):
        pack = KOKORO_VOICE_PACKS[key]
        print(f"  {key}) {pack['display']}  {pack['detail']}")
        print(f"     {pack['voices']}")
        print()

    print()
    print()
    print()
    print()
    print()
    print("=" * width)

    return _get_menu_choice(len(KOKORO_VOICE_PACKS), "Selection; Menu Options = 1-2, Abandon = A: ")


# =============================================================================
# NEW INSTALLATION FLOW FUNCTIONS
# =============================================================================

def _determine_backend_type(backend: str) -> str:
    """Map backend menu selection to BACKEND_TYPE string used by configure.py."""
    info = BACKEND_OPTIONS[backend]
    vulkan_binary = info.get("vulkan_required", False) or "Vulkan" in backend
    vulkan_wheel = info.get("build_flags", {}).get("GGML_VULKAN") == "1"

    if vulkan_binary and vulkan_wheel:
        return "VULKAN_VULKAN"
    elif vulkan_binary:
        return "VULKAN_CPU"
    else:
        return "CPU_CPU"


def download_kokoro_voices(pack_key: str) -> bool:
    """Download Kokoro TTS model and voice files for the selected pack.
    Strategy:
      Phase 1 — Download the main Kokoro model weights via snapshot_download.
                 Avoids running inference just to trigger the download, and gives
                 real progress output via live stdout streaming.
      Phase 2 — Download each voice .pt file explicitly with hf_hub_download.
                 Voice files are ~2 MB each; no inference needed.
       Phase 3 — Warm-up: create KPipeline once to verify the install is functional.
                 Fast (model already on disk) and catches import/path errors early.
    """
    pack = KOKORO_VOICE_PACKS.get(pack_key)
    if not pack:
        print_status("Invalid TTS pack selection ", False)
        return False

    python_exe = str(VENV_DIR / "Scripts" / "python.exe")

    voice_ids = pack["voice_ids"]
    lang_code  = pack["lang_code"]
    hf_cache   = str(BASE_DIR / "data" / "tts_models" / "kokoro" / "hub")

    download_script = f'''
import os, sys, shutil
import traceback

# Clear stale locks that might cause huggingface_hub to hang indefinitely on Windows
locks_dir = os.path.join(r"{hf_cache}", "locks")
if os.path.exists(locks_dir):
    try:
        shutil.rmtree(locks_dir)
        print("[TTS] Cleared stale Hugging Face cache locks.", flush=True)
    except Exception as e:
        print(f"[TTS] Warning: Could not clear cache locks: {{e}}", flush=True)

os.environ["HF_HOME"]                       = r"{hf_cache}"
os.environ["HUGGINGFACE_HUB_CACHE"]         = r"{hf_cache}"
os.environ["CUDA_VISIBLE_DEVICES"]          = " "
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # Enable progress to verify it's not hanging
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"]             = "info" # Show info/warnings (e.g., unauthenticated rate limits)

REPO_ID   = "hexgrad/Kokoro-82M"
VOICE_IDS = {voice_ids!r}
LANG_CODE = "{lang_code}"

# ── Phase 1: model weights ──────────────────────────────────────────
print("[TTS] Phase 1/3 — Downloading Kokoro model weights... ", flush=True)
try:
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        cache_dir=r"{hf_cache}",
        ignore_patterns=["*.md", "*.txt", "*.gitattributes"],
    )
    print(f"[TTS] Model weights downloaded to: {{local_dir}} ", flush=True)
except Exception as e:
    print(f"[TTS] ERROR downloading model weights: {{e}} ", flush=True)
    traceback.print_exc()
    sys.exit(1)

# ── Phase 2: voice .pt files ────────────────────────────────────────
print(f"[TTS] Phase 2/3 — Downloading {{len(VOICE_IDS)}} voice file(s)...", flush=True)
failed_voices = []
try:
    from huggingface_hub import hf_hub_download
    for vid in VOICE_IDS:
        filename = f"voices/{{vid}}.pt"
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                cache_dir=r"{hf_cache}",
            )
            print(f"[TTS]   Voice OK: {{vid}}", flush=True)
        except Exception as e:
            print(f"[TTS]   WARNING: could not download {{vid}}: {{e}}", flush=True)
            failed_voices.append(vid)
except Exception as e:
    print(f"[TTS] ERROR in voice download phase: {{e}}", flush=True)
    traceback.print_exc()
    sys.exit(1)

if failed_voices:
    print(f"[TTS] {{len(failed_voices)}} voice(s) failed: {{failed_voices}}", flush=True)
    sys.exit(1)

# ── Phase 3: ensure package + warm-up ───────────────────────────────
# Models/voices are already on disk. Ensure the *package* is importable in
# this venv (Check/Install can leave it missing after a partial prior run),
# then optionally warm up KPipeline. Package missing is fatal; warm-up
# failure after a successful import is a warning only (weights are ready).
print("[TTS] Phase 3/3 — Ensuring kokoro package + pipeline warm-up...", flush=True)
print(f"[TTS] python={{sys.executable}}", flush=True)

try:
    import kokoro  # noqa: F401
except ImportError:
    print("[TTS] kokoro package not found — installing into this venv...", flush=True)
    import subprocess as _sp
    _r = _sp.run(
        [sys.executable, "-m", "pip", "install", "kokoro>=0.9.4", "--upgrade-strategy", "only-if-needed"],
        capture_output=True, text=True,
    )
    if _r.returncode != 0:
        print(f"[TTS] ERROR: pip install kokoro failed:\\n{{_r.stdout}}\\n{{_r.stderr}}", flush=True)
        sys.exit(1)
    try:
        import kokoro  # noqa: F401
    except ImportError as e:
        print(f"[TTS] ERROR: kokoro still not importable after pip install: {{e}}", flush=True)
        sys.exit(1)
    print("[TTS] kokoro package installed.", flush=True)

try:
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code=LANG_CODE, repo_id=REPO_ID)
    print("[TTS] KPipeline created successfully — Kokoro is ready.", flush=True)
except Exception as e:
    # Weights + voices are already downloaded; runtime can still work.
    print(f"[TTS] WARNING: pipeline warm-up failed (models are on disk): {{e}}", flush=True)
    traceback.print_exc()

print("[TTS] All phases complete.", flush=True)
sys.exit(0)
'''
    # Kokoro is download-only (pip wheel install + HF file downloads), never
    # compiled — so its scratch script belongs under data/temp, not the
    # C:\temp_build short-path staging area reserved for actual builds
    # (CMake/MSVC path-length workaround; see WIN_COMPILE_TEMP).
    script_path = BASE_DIR / "data" / "temp" / "download_kokoro.py"
    try:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(download_script)

        print(f"  Downloading Kokoro TTS ({pack['display']}) — output below: ")
        print(f"  (model ~82 MB + {len(voice_ids)} voice file(s) ~2 MB each) ")
        print()

        # Stream output live — user can see progress and it doesn't look like a hang.
        # Timeout is a wall-clock deadline (30 min) rather than subprocess.run timeout 
        # which would kill a legitimately slow download mid-stream.
        #
        # NOTE: no text=True here — text mode's universal-newline translation
        # converts every lone \r (which tqdm relies on to overwrite its own
        # progress line) into \n, turning one live-updating bar into hundreds
        # of separate printed lines. Read raw bytes and split on \r/\n
        # ourselves instead, preserving whichever terminator was actually sent.
        proc = subprocess.Popen(
            [python_exe, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            bufsize=0,                  # unbuffered — we do our own chunking
        )

        timed_out = False
        deadline  = time.time() + 1800   # 30-minute hard cap

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buf = ""
        while True:
            byte = proc.stdout.read(1)
            if byte:
                buf += decoder.decode(byte)
            elif proc.poll() is not None:
                break

            while True:
                idx_r = buf.find("\r")
                idx_n = buf.find("\n")
                candidates = [i for i in (idx_r, idx_n) if i != -1]
                if not candidates:
                    break
                idx = min(candidates)
                segment, term, buf = buf[:idx], buf[idx], buf[idx + 1:]
                # Suppress pip's "[notice] A new release of pip is available" noise
                if segment.startswith("[notice]") or "new release of pip" in segment:
                    continue
                print(f"  {segment}", end=term, flush=True)

            if time.time() > deadline:
                proc.kill()
                timed_out = True
                break

        proc.wait()
        script_path.unlink(missing_ok=True)

        if timed_out:
            print_status("Kokoro TTS download timed out (>30 min).  "
                         "Check your internet connection and try again. ", False)
            return False

        if proc.returncode == 0:
            print()
            print_status(f"Kokoro TTS installed: {pack['display']} ")
            return True
        else:
            print()
            print_status("Kokoro TTS download failed — see output above for details. ", False)
            return False

    except Exception as e:
        print_status(f"Kokoro TTS download error: {e} ", False)
        script_path.unlink(missing_ok=True)
        return False


def download_embedding_model(model_name: str) -> bool:
    """Download the selected embedding model to local cache.

    Skips the download when the model is already fully cached. Streams
    output live (with Hugging Face progress bars) so slow connections show
    progress instead of appearing to hang, under a 45-minute wall-clock
    deadline rather than a short subprocess timeout."""
    python_exe = str(VENV_DIR / "Scripts" / "python.exe")
    cache_dir = str(BASE_DIR / "data" / "embedding_cache")
    cache_parent = str(BASE_DIR / "data")

    script = f'''
import os, sys, traceback
os.environ["TRANSFORMERS_CACHE"] = r"{cache_dir}"
os.environ["HF_HOME"] = r"{cache_parent}"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"{cache_dir}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"   # show download progress
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL = "{model_name}"

def finalize(model):
    """Run a test encode to ensure all runtime files are loaded/cached."""
    print("[EMBED] Running test encoding to finalize cache...", flush=True)
    test = model.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
    print(f"[EMBED] Model fully initialized and cached (dim={{test.shape[1]}})", flush=True)

# ── Phase 1: check local cache (offline) ────────────────────────────
print(f"[EMBED] Checking local cache for: {{MODEL}}", flush=True)
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
except Exception as e:
    print(f"[EMBED] Error importing libraries: {{e}}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    model = SentenceTransformer(MODEL, cache_folder=r"{cache_dir}", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_folder=r"{cache_dir}", local_files_only=True)
    finalize(model)
    print("[EMBED] Already cached - download skipped.", flush=True)
    sys.exit(0)
except Exception:
    print("[EMBED] Not cached (or cache incomplete) - downloading...", flush=True)

# ── Phase 2: download (progress bars stream to console) ─────────────
try:
    print(f"[EMBED] Downloading SentenceTransformer model: {{MODEL}}", flush=True)
    model = SentenceTransformer(MODEL, cache_folder=r"{cache_dir}")

    print("[EMBED] Downloading and caching tokenizer files...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_folder=r"{cache_dir}")

    finalize(model)
    sys.exit(0)
except Exception as e:
    print(f"[EMBED] Error: {{e}}", flush=True)
    traceback.print_exc()
    sys.exit(1)
'''
    script_path = TEMP_DIR / "download_embedding.py"
    try:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

        print_status(f"Preparing embedding model: {model_name} — output below:")
        print("  (bge-base is ~430 MB; progress shown live, 45-minute limit)")
        print()

        # Stream output live — user sees HF progress bars instead of a hang.
        # Timeout is a wall-clock deadline rather than subprocess.run timeout
        # which would kill a legitimately slow download mid-stream.
        proc = subprocess.Popen(
            [python_exe, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # HF progress bars print to stderr
            text=True,
            bufsize=1,                  # line-buffered
        )

        timed_out = False
        deadline  = time.time() + 2700   # 45-minute hard cap

        # Forward char-by-char so tqdm progress bars (which update with
        # carriage returns, not newlines) render live instead of buffering.
        while True:
            ch = proc.stdout.read(1)
            if ch == "":
                if proc.poll() is not None:
                    break
                continue
            sys.stdout.write(ch)
            sys.stdout.flush()
            if time.time() > deadline:
                proc.kill()
                timed_out = True
                break

        proc.wait()
        script_path.unlink(missing_ok=True)

        if timed_out:
            print_status("Embedding model download timed out (>45 min). "
                         "Check your internet connection and try again — "
                         "already-downloaded files are kept and will be reused.", False)
            return False

        if proc.returncode == 0:
            print()
            print_status(f"Embedding model installed: {model_name}")
            return True
        else:
            print()
            print_status("Embedding model download failed — see output above for details.", False)
            return False

    except Exception as e:
        print_status(f"Embedding model download error: {e}", False)
        script_path.unlink(missing_ok=True)
        return False


def create_config_jsons():
    """Create data/configuration.json and data/preferences.json with defaults.

    Existing values are preserved, so re-running the installer over a working
    install does not wipe the user's settings; only missing keys are filled in.

    These two dictionaries are duplicated from scripts/configure.py
    (CONFIGURATION_DEFAULTS / PREFERENCES_DEFAULTS). The installer runs on the
    system Python before the venv exists, so it cannot import that module.
    If a default changes there, change it here too.
    """
    configuration_defaults = {
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

    preferences_defaults = {
        # Program options
        "session_log_height": 625,
        "max_attach_slots": 6,
        "max_history_slots": 12,
        # Output options
        "show_think_phase": False,
        "bleep_on_events": False,
        "print_raw_output": False,
        # Filter settings, as [find, replace] pairs
        "filter_rules": [
            ["\r\n", "\n"],
            ["\r", "\n"],
            ["###### ", ""],
            ["##### ", ""],
            ["#### ", ""],
            ["### ", ""],
            ["## ", ""],
            ["# ", ""],
            ["**", ""],
            ["__", ""],
            ["- ", "* "],
            ["* ", "* "],
        ],
    }

    def write_settings(filename, section, defaults):
        path = BASE_DIR / "data" / filename
        values = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict) and isinstance(existing.get(section), dict):
                    values = existing[section]
            except Exception:
                print_status(f"{filename} unreadable — rewriting from defaults", False)
                values = {}

        for key, default_val in defaults.items():
            if key not in values:
                values[key] = default_val

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({section: values}, f, indent=4, ensure_ascii=False)
        print_status(f"Settings file created: data/{filename}")

    write_settings("configuration.json", "configuration", configuration_defaults)
    write_settings("preferences.json", "preferences", preferences_defaults)

def refresh_configs():
    """Only regenerate INI/JSON files without reinstalling packages."""
    import configparser as _cp

    print_status("Refreshing configuration files...")

    ini_path = BASE_DIR / "data" / "constants.ini"
    if not ini_path.exists():
        print_status("No existing constants.ini found — cannot refresh", False)
        return

    try:
        cfg_ini = _cp.ConfigParser()
        cfg_ini.read(ini_path, encoding='utf-8')

        system = cfg_ini['system']

        os_version        = system.get('os_version', 'unknown')
        python_version    = system.get('python_version', f"{sys.version_info.major}.{sys.version_info.minor}")
        backend_type      = system.get('backend_type', 'CPU_CPU')
        embedding_model   = system.get('embedding_model', 'BAAI/bge-small-en-v1.5')
        vulkan_available  = system.getboolean('vulkan_available', False)
        llama_cli_path    = system.get('llama_cli_path', None)
        llama_bin_path    = system.get('llama_bin_path', None)
        dx_feature_level  = system.getint('dx_feature_level', 0)
        browser_acceleration = system.getboolean('browser_acceleration', True)

        # TTS settings
        tts_section = cfg_ini['tts'] if 'tts' in cfg_ini else {}
        tts_pack = int(tts_section.get('tts_pack', '1'))
        tts_default_voice_id = tts_section.get('tts_default_voice_id', 'af_heart')
        tts_default_voice_name = tts_section.get('tts_default_voice_name', 'Heart — American Female')
        tts_enabled_str = tts_section.get('tts_enabled_voices', '')
        tts_enabled_voices = [v.strip() for v in tts_enabled_str.split(',') if v.strip()]

    except Exception as e:
        print_status(f"Could not read existing INI: {e}", False)
        return

    # Recreate INI with single os_version key
    create_system_ini(
        os_version=os_version,
        python_version=python_version,
        backend_type=backend_type,
        embedding_model=embedding_model,
        vulkan_available=vulkan_available,
        llama_cli_path=llama_cli_path,
        llama_bin_path=llama_bin_path,
        tts_engine="kokoro",
        tts_pack=tts_pack,
        tts_default_voice_id=tts_default_voice_id,
        tts_default_voice_name=tts_default_voice_name,
        tts_enabled_voices=tts_enabled_voices,
        browser_acceleration=browser_acceleration,
        dx_feature_level=dx_feature_level,
    )

    create_config_jsons()
    print_status("Configuration files refreshed successfully")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def _read_install_record() -> dict:
    """Read [install] + relevant [system] fields from constants.ini if present."""
    import configparser as _cp
    ini_path = BASE_DIR / "data" / "constants.ini"
    out = {}
    if not ini_path.exists():
        return out
    try:
        cfg = _cp.ConfigParser()
        cfg.read(ini_path, encoding='utf-8')
        if 'system' in cfg:
            for k in ('backend_type', 'embedding_model', 'llama_cli_path',
                      'llama_bin_path', 'llama_wheel_version'):
                if k in cfg['system']:
                    out[k] = cfg['system'].get(k)
        if 'install' in cfg:
            for k, v in cfg['install'].items():
                out[k] = v
        if 'tts' in cfg:
            out['tts_pack'] = cfg['tts'].get('tts_pack')
            out['tts_default_voice_id'] = cfg['tts'].get('tts_default_voice_id')
            out['tts_enabled_voices'] = cfg['tts'].get('tts_enabled_voices', '')
    except Exception as e:
        print(f"[INI] Could not read install record: {e}")
    return out


def _binary_dir_looks_vulkan(bin_dir: Path) -> bool | None:
    """True if dir has Vulkan llama bits, False if CPU-only, None if empty/missing."""
    if not bin_dir.is_dir():
        return None
    names = [p.name.lower() for p in bin_dir.iterdir() if p.is_file()]
    if not names:
        return None
    has_cli = any(n == 'llama-cli.exe' or n.startswith('llama-cli') for n in names)
    if not has_cli:
        return None
    vk_markers = ('ggml-vulkan', 'vulkan-1', 'libvulkan')
    if any(any(m in n for m in vk_markers) for n in names):
        return True
    return False


def _ensure_prebuilt_binary(backend: str) -> bool:
    """Download/extract prebuilt llama.cpp zip when BACKEND_OPTIONS provides a URL."""
    info = BACKEND_OPTIONS[backend]
    url = info.get('url')
    dest = info.get('dest')
    if not url or not dest:
        return True

    dest_path = BASE_DIR / dest
    cli = BASE_DIR / (info.get('cli_path') or '')
    want_vulkan = 'vulkan' in backend.lower() or bool(info.get('vulkan_required'))

    existing = _binary_dir_looks_vulkan(dest_path)
    if cli.is_file() and existing is not None:
        if want_vulkan and existing is True:
            print_status(f"Prebuilt Vulkan binary already present: {dest}")
            return True
        if not want_vulkan and existing is False:
            print_status(f"Prebuilt CPU binary already present: {dest}")
            return True
        print_status(f"Existing binary at {dest} does not match required type — re-downloading")
        _force_rmtree(dest_path)

    print_status(f"Downloading prebuilt binary: {url}")
    try:
        import urllib.request
        dest_path.mkdir(parents=True, exist_ok=True)
        zip_path = TEMP_DIR / "llama_prebuilt.zip"
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(zip_path))
        import zipfile
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(dest_path))
        zip_path.unlink(missing_ok=True)
        children = list(dest_path.iterdir())
        if len(children) == 1 and children[0].is_dir():
            nested = children[0]
            for item in nested.iterdir():
                target = dest_path / item.name
                if target.exists():
                    if target.is_dir():
                        _force_rmtree(target)
                    else:
                        target.unlink()
                item.rename(target)
            nested.rmdir()
        if not (BASE_DIR / info['cli_path']).is_file():
            found = list(dest_path.rglob('llama-cli.exe'))
            if found:
                print_status(f"Found llama-cli.exe at {found[0].relative_to(BASE_DIR)}")
            else:
                print_status("Downloaded archive but llama-cli.exe not found", False)
                return False
        print_status(f"Prebuilt binary installed to {dest}")
        return True
    except Exception as e:
        print_status(f"Prebuilt binary download failed: {e}", False)
        return False


def _ensure_llama_wheel_for_backend(backend: str, force: bool = False) -> bool:
    """Ensure llama-cpp-python wheel matches the backend (CPU vs Vulkan)."""
    global _INSTALLED_LLAMA_WHEEL_VERSION
    info = BACKEND_OPTIONS[backend]
    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
    needs_vulkan = bool(info.get("build_flags", {}).get("GGML_VULKAN")) or (
        info.get("vulkan_required") and info.get("compile_wheel")
    )
    compile_wheel = bool(info.get("compile_wheel"))

    existing = get_installed_llama_info()
    if existing and not force:
        if not compile_wheel and not needs_vulkan and not existing["vulkan"]:
            print_status(f"llama-cpp-python {existing['version']} (CPU) already matches — skipping")
            _INSTALLED_LLAMA_WHEEL_VERSION = f"v{existing['version']}"
            return True
        if compile_wheel and existing["vulkan"] == needs_vulkan:
            kind = "Vulkan" if needs_vulkan else "CPU"
            print_status(f"llama-cpp-python {existing['version']} ({kind}) already matches — skipping")
            _INSTALLED_LLAMA_WHEEL_VERSION = f"v{existing['version']}"
            return True

    if not compile_wheel:
        wheel_version = LLAMACPP_PYTHON_PREBUILT_VERSION.lstrip("v")
        if existing and existing["version"] == wheel_version and not existing["vulkan"] and not force:
            _INSTALLED_LLAMA_WHEEL_VERSION = f"v{wheel_version}"
            print_status(f"llama-cpp-python {wheel_version} (CPU) already installed — skipping")
            return True
        sources = _get_prebuilt_wheel_urls()
        if not sources:
            print_status("No pre-built wheel sources available.", False)
            return False
        print_status(f"Installing llama-cpp-python {wheel_version} (CPU)...")
        installed = False
        for src in sources:
            label = src.get("label", src["value"])
            print(f"  Trying: {label}")
            if src["type"] == "url":
                installed = pip_install_with_retry(pip_exe, src["value"], max_retries=2, initial_delay=3.0)
            elif src["type"] == "pypi":
                installed = pip_install_with_retry(
                    pip_exe, src["value"],
                    extra_args=["--prefer-binary"],
                    max_retries=3, initial_delay=5.0,
                )
            if installed:
                print_status(f"llama-cpp-python {wheel_version} installed via {label}")
                _INSTALLED_LLAMA_WHEEL_VERSION = f"v{wheel_version}"
                break
            print(f"  Source unavailable: {label}")
        if not installed:
            print_status(f"llama-cpp-python {wheel_version} could not be installed.", False)
            return False
        return True

    build_flags = info.get("build_flags", {})
    if build_flags.get("GGML_VULKAN") and not check_vulkan_sdk_installed():
        print_status("Error: Vulkan SDK not found (required for Vulkan wheel)", False)
        return False
    if not check_vcredist_windows():
        print_status("Warning: Visual C++ Redistributable (x64) not detected", False)
        time.sleep(2)
    if not build_llama_cpp_python_with_flags(build_flags):
        return False
    _INSTALLED_LLAMA_WHEEL_VERSION = LLAMACPP_PYTHON_VERSION
    return True


def _finalize_configs(backend: str, embedding_model: str, tts_choice: str,
                      voice_pack: dict, install_method: str) -> None:
    """Write constants.ini + JSON configs after a successful install step."""
    info = BACKEND_OPTIONS[backend]
    backend_type = _determine_backend_type(backend)
    wheel_info = get_installed_llama_info()
    llama_wheel_vulkan = bool(wheel_info and wheel_info.get("vulkan"))

    create_system_ini(
        os_version=f"Windows {WINDOWS_VERSION}" if WINDOWS_VERSION else "unknown",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        backend_type=backend_type,
        embedding_model=embedding_model,
        vulkan_available=_DETECTED_VULKAN,
        llama_cli_path=info.get("cli_path"),
        llama_bin_path=info.get("dest"),
        tts_engine="kokoro",
        tts_pack=int(tts_choice) if str(tts_choice).isdigit() else 1,
        tts_default_voice_id=voice_pack.get("default_voice_id") if voice_pack else None,
        tts_default_voice_name=voice_pack.get("default_voice_name") if voice_pack else None,
        tts_enabled_voices=voice_pack.get("voice_ids") if voice_pack else None,
        browser_acceleration=_DETECTED_DX_CAPABLE,
        dx_feature_level=_DETECTED_DX_LEVEL,
        install_method=install_method,
        backend_key=backend,
        llama_wheel_vulkan=llama_wheel_vulkan,
    )
    if _INSTALLED_LLAMA_WHEEL_VERSION:
        update_ini_wheel_version(_INSTALLED_LLAMA_WHEEL_VERSION)
    create_config_jsons()


def run_installer():
    """Main installer flow with 4-option menu.

    1. Clean/Purge Install All
    2. Check/Install Python/Libraries
    3. Check/Recompile Binaries
    4. Refresh Configs/Inis
    """
    if not check_version_compatibility():
        print("\nSystem requirements not met. Installation cannot continue.")
        return

    run_detections_once()
    detect_windows_version()

    main_choice = show_main_menu()
    if main_choice == 'A':
        print("\nInstallation abandoned.")
        return

    if main_choice == '4':
        refresh_configs()
        return

    is_clean = (main_choice == '1')
    libraries_only = (main_choice == '2')
    binaries_only = (main_choice == '3')

    record = _read_install_record()

    backend_choice = show_backend_menu()
    if backend_choice == 'A':
        print("\nInstallation abandoned.")
        return
    backend_keys = list(BACKEND_OPTIONS.keys())
    backend = backend_keys[int(backend_choice) - 1]

    embedding_model = record.get('embedding_model') or EMBEDDING_MODELS['1']['name']
    tts_choice = record.get('tts_pack') or '1'
    voice_pack = KOKORO_VOICE_PACKS.get(str(tts_choice), KOKORO_VOICE_PACKS['1'])

    if not binaries_only:
        embed_choice = show_embedding_menu()
        if embed_choice == 'A':
            print("\nInstallation abandoned.")
            return
        embedding_model = EMBEDDING_MODELS[embed_choice]['name']

        tts_choice = show_tts_menu()
        if tts_choice == 'A':
            print("\nInstallation abandoned.")
            return
        voice_pack = KOKORO_VOICE_PACKS[tts_choice]

    clear_screen()
    print_header("Installing...")

    if is_clean:
        print_status("Starting Clean/Purge Install All...")
        if VENV_DIR.exists():
            _force_rmtree(VENV_DIR)
            print_status("Removed existing virtual environment")
    elif libraries_only:
        print_status("Starting Check/Install Python/Libraries...")
    else:
        print_status("Starting Check/Recompile Binaries...")

    create_files_and_directories(backend)

    if is_clean:
        if not create_venv():
            return
    else:
        if not ensure_venv():
            return

    if binaries_only:
        if not _ensure_prebuilt_binary(backend):
            print_status("Binary setup failed. Installation aborted.", False)
            return
        if not _ensure_llama_wheel_for_backend(backend, force=False):
            print_status("llama-cpp-python setup failed. Installation aborted.", False)
            return
        _finalize_configs(backend, embedding_model, str(tts_choice), voice_pack, "binaries")
        print()
        print("=" * (shutil.get_terminal_size().columns - 1))
        print_status("Binary check/recompile complete!")
        print("=" * (shutil.get_terminal_size().columns - 1))
        print()
        return

    if not install_python_deps(backend, skip_if_present=not is_clean):
        print_status("Python dependency installation failed. Installation aborted.", False)
        return

    if not install_optional_file_support():
        print_status("Optional file support installation failed. Installation aborted.", False)
        return

    if not download_embedding_model(embedding_model):
        print_status("Embedding model download failed. Installation aborted.", False)
        return

    if not download_kokoro_voices(tts_choice):
        print_status("Kokoro TTS download failed. Installation aborted.", False)
        return

    if not _ensure_prebuilt_binary(backend):
        print_status("Binary setup failed. Installation aborted.", False)
        return

    method = "clean" if is_clean else "libraries"
    _finalize_configs(backend, embedding_model, str(tts_choice), voice_pack, method)

    print()
    print("=" * (shutil.get_terminal_size().columns - 1))
    print_status("Installation complete!")
    print("=" * (shutil.get_terminal_size().columns - 1))
    print()
    print("  You can now run the application using the launcher.")
    print()



if __name__ == "__main__":
    snapshot_pre_existing_processes()
    atexit.register(cleanup_build_processes)
    try:
        run_installer()
    except KeyboardInterrupt:
        print("\n\nInstallation interrupted by user.")
    except Exception as e:
        print(f"\n\nInstallation failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_build_processes()