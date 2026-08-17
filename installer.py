# Script: installer.py - Installation script for Chat-Gradio-Gguf
# v2: Targets Windows 10-11 / Ubuntu 24-25 / Python 3.11-3.12 / Gradio 5.x / PyQt6
# Note: Uses sentence-transformers for embeddings (cross-platform, offline)
# Note: Uses PyQt6 WebEngine for custom browser window

import os
import json
import subprocess
import sys
import contextlib
import time
import threading
from pathlib import Path
import shutil
import atexit
import re

# Constants / Variables
_PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
APP_NAME = "Chat-Gradio-Gguf"
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
LINUX_COMPILE_TEMP = None
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

PLATFORM = None

def set_platform() -> None:
    global PLATFORM
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
        print("ERROR: Platform argument required (windows/linux)")
        sys.exit(1)
    PLATFORM = sys.argv[1].lower()

set_platform()

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

    if PLATFORM == "windows":
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

    else:  # Linux
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read().lower()
            features["AVX"]    = 'avx'    in content
            features["AVX2"]   = 'avx2'   in content
            features["AVX512"] = 'avx512' in content
            features["FMA"]    = 'fma'    in content
            features["F16C"]   = 'f16c'   in content
            features["SSE3"]   = 'sse3'   in content or 'pni' in content
            features["SSSE3"]  = 'ssse3'  in content
            features["SSE4_1"] = 'sse4_1' in content
            features["SSE4_2"] = 'sse4_2' in content
            success = True
        except Exception:
            features["SSE3"] = True
            print_status("Linux CPU detection fallback - SSE3 only")
            success = True

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

    if PLATFORM != "windows":
        DX11_CAPABLE = True
        DX_FEATURE_LEVEL = 0xb000
        DX_FEATURE_NAME = "11.0"
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


# Set TEMP_DIR based on platform
if PLATFORM == "windows":
    TEMP_DIR = WIN_COMPILE_TEMP
else:
    TEMP_DIR = BASE_DIR / "data" / "temp"

# Backend definitions
if PLATFORM == "windows":
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
else:  # Linux
    BACKEND_OPTIONS = {
        "Download CPU Binary / Default CPU Wheel": {
            "url": None, "dest": None, "cli_path": None,
            "needs_python_bindings": True, "compile_binary": False,
            "compile_wheel": False, "vulkan_required": False, "build_flags": {}
        },
        "Download Vulkan Binary / Default CPU Wheel": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-ubuntu-vulkan-x64.tar.gz",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",
            "needs_python_bindings": True, "compile_binary": False,
            "compile_wheel": False, "vulkan_required": False, "build_flags": {}
        },
        "Compile CPU Binaries / Compile CPU Wheel": {
            "url": None, "dest": "data/llama-cpu-bin",
            "cli_path": "data/llama-cpu-bin/llama-cli",
            "needs_python_bindings": True, "compile_binary": True,
            "compile_wheel": True, "vulkan_required": False, "build_flags": {}
        },
        "Compile Vulkan Binaries / Compile Vulkan Wheel": {
            "url": None, "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",
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
]

if PLATFORM == "windows":
    BASE_REQ.extend([
        "pywin32>=306",
        "tk==0.1.0",
        "pythonnet==3.0.5",
    ])

def clear_screen():
    os.system('cls' if PLATFORM == "windows" else 'clear')

def backend_requires_compilation(backend: str) -> bool:
    """Check if the selected backend requires compilation"""
    info = BACKEND_OPTIONS.get(backend, {})
    return info.get("compile_binary", False) or info.get("compile_wheel", False)


def detect_windows_version() -> str:
    """Detect Windows version and cache in global."""
    global WINDOWS_VERSION
    if WINDOWS_VERSION is not None:
        return WINDOWS_VERSION

    if PLATFORM != "windows":
        return None

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


def detect_linux_version() -> str:
    """Detect Ubuntu version and cache in global OS_VERSION."""
    global OS_VERSION
    if OS_VERSION is not None:
        return OS_VERSION

    try:
        with open("/etc/os-release") as f:
            content = f.read()

        if "ubuntu" not in content.lower():
            OS_VERSION = "unknown"
            return "unknown"

        version_match = re.search(r'VERSION_ID="?([0-9\.]+)"?', content)
        if version_match:
            OS_VERSION = version_match.group(1)
            return OS_VERSION

        OS_VERSION = "unknown"
        return "unknown"
    except Exception as e:
        print_status(f"OS detection failed: {e}", False)
        OS_VERSION = "unknown"
        return "unknown"

def get_dynamic_requirements() -> list:
    requirements = BASE_REQ.copy()
    requirements.append("gradio>=5.0.0,<6.0.0")
    return requirements


def get_torch_version_for_python() -> str:
    return "torch>=2.5.0"

def check_version_compatibility():
    """Check Python and OS compatibility."""
    global WINDOWS_VERSION, PYTHON_VERSION, PLATFORM

    if sys.version_info < (3, 11):
        print_status("Python ≥3.11 required for Chat-Gradio-Gguf v2", False)
        return False

    if sys.version_info >= (3, 13):
        print_status(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported. "
            "Kokoro TTS requires Python <3.13. Please use Python 3.11 or 3.12.", False
        )
        return False

    PYTHON_VERSION = sys.version_info

    if PLATFORM == "windows":
        win_ver = detect_windows_version()
        if win_ver in ("unsupported", "7", "8", "8.1"):
            print_status(
                f"Windows {win_ver} is not supported in v2. Requires Windows 10 or 11.", False
            )
            return False
        return True

    else:  # Linux
        try:
            with open("/etc/os-release") as f:
                content = f.read()
            if "UBUNTU_VERSION_ID" in content or "ubuntu" in content.lower():
                version_match = re.search(r'VERSION_ID="?([0-9\.]+)"?', content)
                if version_match:
                    ubuntu_version = version_match.group(1)
                    major = int(ubuntu_version.split('.')[0])
                    if major >= 24:
                        return True
                    print_status(
                        f"Ubuntu {ubuntu_version} is not supported in v2. Requires Ubuntu 24 or 25.", False
                    )
                    return False
            print_status("Could not determine Ubuntu version", False)
            return False
        except Exception as e:
            print_status(f"OS detection failed: {e}", False)
            return False


def is_kokoro_compatible() -> bool:
    """Check if current OS/Python supports Kokoro TTS."""
    if sys.version_info >= (3, 13):
        return False

    if PLATFORM == "windows":
        return WINDOWS_VERSION in ["10", "11"]

    elif PLATFORM == "linux":
        try:
            if OS_VERSION:
                major_version = int(OS_VERSION.split('.')[0])
                return major_version >= 24
        except (ValueError, AttributeError, IndexError):
            pass
        return False

    return False


# =============================================================================
# INSTALLATION HELPERS
# =============================================================================

def snapshot_pre_existing_processes() -> None:
    global _PRE_EXISTING_PROCESSES
    if PLATFORM != "windows":
        return
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
    if PLATFORM != "windows":
        return
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
    elif PLATFORM == "windows":
        # Priority 2: bundled inside VS / Build Tools 2019-2022
        cmake_bin = _find_cmake_in_vs_installations()
        if cmake_bin:
            tools["CMake"] = True
            # Prepend to PATH so cmake is usable by any subsequent subprocess
            os.environ["PATH"] = cmake_bin + os.pathsep + os.environ.get("PATH", "")

    # MSVC / MSBuild (Windows only) ----------------------------------------
    if PLATFORM == "windows":
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
    print(f"    Chat-Gradio-Gguf v2 — {section}")
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
    # BASE_DIR, and on Windows TEMP_DIR is C:/temp_build (WIN_COMPILE_TEMP,
    # kept short so build paths do not blow the path length limit). Nothing
    # created it, which is why the first step to write a temp script there
    # failed with "No such file or directory". On Linux TEMP_DIR is
    # data/temp and already in the list, so this is a no-op there.
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    print_status(f"Directories created/verified (temp: {TEMP_DIR})")


# =============================================================================
# EMBEDDING BACKEND INSTALLATION
# =============================================================================

def install_embedding_backend() -> bool:
    """Install PyTorch CPU and sentence-transformers."""
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe    = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("pip.exe"    if PLATFORM == "windows" else "pip"))

    torch_req                   = "torch>=2.5.0"
    transformers_version        = "transformers>=4.44.0"
    sentence_transformers_version = "sentence-transformers>=3.3.0"

    print_status(f"Installing PyTorch (CPU) — {torch_req}...")
    if not pip_install_with_retry(pip_exe, torch_req,
                                  ["--index-url", "https://download.pytorch.org/whl/cpu",
                                   "--upgrade-strategy", "only-if-needed"],
                                  max_retries=10, initial_delay=5.0):
        print_status("PyTorch installation failed", False)
        return False
    print_status("PyTorch (CPU) installed")

    subprocess.run(
        [pip_exe, "install", "setuptools>=80.0", "--upgrade", "--quiet"],
        capture_output=True, timeout=120
    )
    print_status("setuptools restored after torch install")

    print_status(f"Installing {transformers_version}...")
    if not pip_install_with_retry(pip_exe, transformers_version,
                                  ["--upgrade-strategy", "only-if-needed"],
                                  max_retries=10, initial_delay=5.0):
        print_status("transformers installation failed", False)
        return False
    print_status("transformers installed")

    print_status(f"Installing {sentence_transformers_version}...")
    if not pip_install_with_retry(pip_exe, sentence_transformers_version,
                                  ["--upgrade-strategy", "only-if-needed"],
                                  max_retries=10, initial_delay=5.0):
        print_status("sentence-transformers installation failed", False)
        return False
    print_status("sentence-transformers installed")

    verify_script = '''
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    import torch
    torch.set_grad_enabled(False)
    print(f"torch: {torch.__version__}")
    from sentence_transformers import SentenceTransformer
    print("sentence_transformers: OK")
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
'''
    verify_path = TEMP_DIR / "verify_embedding.py"
    try:
        verify_path.parent.mkdir(parents=True, exist_ok=True)
        with open(verify_path, 'w', encoding='utf-8') as f:
            f.write(verify_script)
        verify_result = subprocess.run(
            [python_exe, str(verify_path)], capture_output=True, text=True, timeout=120
        )
        verify_path.unlink(missing_ok=True)

        if verify_result.returncode == 0 and "SUCCESS" in verify_result.stdout:
            print_status("Embedding backend verified")
            return True
        else:
            print_status("Embedding backend verification failed", False)
            print(f"Output: {verify_result.stdout}")
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
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                 ("pip.exe" if PLATFORM == "windows" else "pip"))

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
# LINUX SYSTEM DEPENDENCIES
# =============================================================================

def install_linux_system_dependencies(backend: str) -> bool:
    """Install Linux system dependencies."""
    print_status("Installing Linux system dependencies...")

    base_packages = [
        "python3-dev", "build-essential", "libffi-dev", "libssl-dev",
        "libegl1", "libgl1", "libxkbcommon0", "libxkbcommon-x11-0",
        "libxcb-cursor0", "libxcb-icccm4", "libxcb-image0", "libxcb-keysyms1",
        "libxcb-randr0", "libxcb-render-util0", "libxcb-shape0",
        "libxcb-xinerama0", "libxcb-xkb1", "libxcb1",
    ]

    info = BACKEND_OPTIONS[backend]
    vulkan_packages = []
    if info.get("build_flags", {}).get("GGML_VULKAN"):
        vulkan_packages = [
            "vulkan-tools", "libvulkan-dev", "mesa-utils",
            "glslang-tools", "spirv-tools"
        ]
    elif "Vulkan" in backend:
        vulkan_packages = ["vulkan-tools", "libvulkan1"]

    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(
            ["sudo", "apt-get", "install", "-y"] + list(set(base_packages)), check=True
        )
        print_status("Base dependencies installed")

        if vulkan_packages:
            print_status("Installing Vulkan dependencies...")
            for package in vulkan_packages:
                try:
                    subprocess.run(
                        ["sudo", "apt-get", "install", "-y", package],
                        capture_output=True, check=True
                    )
                    print_status(f"  Installed {package}")
                except subprocess.CalledProcessError:
                    print_status(f"  Optional package {package} failed (continuing)", False)

        print_status("Linux system dependencies installed")
        return True

    except subprocess.CalledProcessError as e:
        print_status(f"System dependency installation failed: {e}", False)
        return False


# =============================================================================
# PYTHON DEPENDENCY INSTALLATION
# =============================================================================

def install_python_deps(backend: str) -> bool:
    """Install Python dependencies."""
    global _INSTALLED_LLAMA_WHEEL_VERSION
    print_status("Installing Python dependencies...")
    try:
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("pip.exe" if PLATFORM == "windows" else "pip"))

        all_requirements = get_dynamic_requirements()

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

        if not info.get("compile_wheel"):
            wheel_version = LLAMACPP_PYTHON_PREBUILT_VERSION.lstrip("v")
            sources = _get_prebuilt_wheel_urls()

            if not sources:
                print_status("No pre-built wheel sources available for this platform.", False)
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

            if build_flags.get("GGML_VULKAN"):
                print_status("Vulkan wheel build - checking Vulkan SDK...")
                if not check_vulkan_sdk_installed():
                    print_status("Error: Vulkan SDK not found", False)
                    return False

            if PLATFORM == "windows" and not check_vcredist_windows():
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

    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                 ("pip.exe" if PLATFORM == "windows" else "pip"))

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

def create_system_ini(platform, os_version, python_version,
                      backend_type, embedding_model,
                      windows_version=None, vulkan_available=False,
                      llama_cli_path=None, llama_bin_path=None,
                      tts_engine="kokoro", 
                      tts_pack=1, tts_default_voice_id=None,
                      tts_default_voice_name=None, tts_enabled_voices=None,
                      browser_acceleration=True,
                      dx_feature_level=0):
    system_ini_path = BASE_DIR / "data" / "constants.ini"
    try:
        with open(system_ini_path, "w", encoding='utf-8') as f:
            f.write("[system]\n")
            f.write(f"platform = {platform}\n")
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
            if platform == "windows" and windows_version:
                f.write(f"windows_version = {windows_version}\n")

            f.write("\n[tts]\n")
            f.write(f"tts_type = {tts_engine}\n")
            f.write(f"tts_pack = {tts_pack}\n")
            f.write(f"tts_default_voice_id = {tts_default_voice_id or 'af_heart'}\n")
            f.write(f"tts_default_voice_name = {tts_default_voice_name or 'Heart — American Female'}\n")
            if tts_enabled_voices:
                f.write(f"tts_enabled_voices = {','.join(tts_enabled_voices)}\n")
            else:
                # fallback: all voices from all packs
                all_ids = []
                for pack in KOKORO_VOICE_PACKS.values():
                    all_ids.extend(pack["voice_ids"])
                f.write(f"tts_enabled_voices = {','.join(all_ids)}\n")

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

        python_exe = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / \
                     ("python.exe" if PLATFORM == "windows" else "python")
        pip_exe    = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / \
                     ("pip.exe"    if PLATFORM == "windows" else "pip")

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
    if PLATFORM == "windows":
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
    else:
        try:
            result = subprocess.run(["vulkaninfo", "--summary"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                return False
            result = subprocess.run(["which", "glslc"], capture_output=True)
            return result.returncode == 0
        except:
            return False


def is_vulkan_installed() -> bool:
    """Check if Vulkan runtime is installed on the system."""
    if PLATFORM == "windows":
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

    else:
        try:
            result1 = subprocess.run(["vulkaninfo", "--summary"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result2 = subprocess.run(["ldconfig", "-p"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return result1.returncode == 0 or b"libvulkan" in result2.stdout
        except FileNotFoundError:
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

    if PLATFORM == "windows":
        filename = f"llama_cpp_python-{wheel_version}-{py_tag}-{py_tag}-win_amd64.whl"
        sources.append({
            "type": "url",
            "label": f"eswarthammana/llama-cpp-wheels {wheel_version}",
            "value": f"https://github.com/eswarthammana/llama-cpp-wheels/releases/download/{LLAMACPP_PYTHON_PREBUILT_VERSION}/{filename}"
        })
    else:
        filename = f"llama_cpp_python-{wheel_version}-{py_tag}-{py_tag}-manylinux_2_31_x86_64.manylinux_2_17_x86_64.whl"
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
    if PLATFORM == "windows" and not shutil.which("cmake"):
        cmake_bin = _find_cmake_in_vs_installations()
        if cmake_bin:
            os.environ["PATH"] = cmake_bin + os.pathsep + os.environ.get("PATH", "")
            print_status(f"CMake located at: {cmake_bin}")
    
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                 ("pip.exe" if PLATFORM == "windows" else "pip"))
    
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
    print()
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
    if PLATFORM == "windows":
        win_ver = WINDOWS_VERSION or detect_windows_version() or "unknown"
        plat_str = f"Windows {win_ver}"
    else:
        linux_ver = OS_VERSION or detect_linux_version() or "unknown"
        plat_str = f"Ubuntu {linux_ver}"
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
    print("   1. Clean Install (Purge First)")
    print()
    print("   2. Check/Install (Fix Missing Packages/Libraries)")
    print()
    print("   3. Refresh Configs (Only Remake Ini/Json)")
    print()
    print()
    print("=" * width)

    return _get_menu_choice(3, "Selection; Menu Options = 1-3, Abandon Install = A: ")


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

    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("python.exe" if PLATFORM == "windows" else "python"))

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

# ── Phase 3: warm-up / verify ───────────────────────────────────────
print("[TTS] Phase 3/3 — Verifying Kokoro pipeline (warm-up)...", flush=True)
try:
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code=LANG_CODE, repo_id=REPO_ID)
    print("[TTS] KPipeline created successfully — Kokoro is ready.", flush=True)
except Exception as e:
    print(f"[TTS] ERROR during pipeline warm-up: {{e}}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("[TTS] All phases complete.", flush=True)
sys.exit(0)
'''
    script_path = TEMP_DIR / "download_kokoro.py"
    try:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(download_script)

        print(f"  Downloading Kokoro TTS ({pack['display']}) — output below: ")
        print(f"  (model ~82 MB + {len(voice_ids)} voice file(s) ~2 MB each) ")
        print()

        # Stream output live — user can see progress and it doesn't look like a hang.
        # Timeout is a wall-clock deadline (15 min) rather than subprocess.run timeout 
        # which would kill a legitimately slow download mid-stream.
        proc = subprocess.Popen(
            [python_exe, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            text=True,
            bufsize=1,                  # line-buffered
        )

        timed_out = False
        deadline  = time.time() + 900   # 15-minute hard cap

        for line in proc.stdout:
            # Suppress pip's "[notice] A new release of pip is available" noise
            if line.startswith("[notice]") or "new release of pip" in line:
                continue
            print(f"  {line}", end="", flush=True)
            if time.time() > deadline:
                proc.kill()
                timed_out = True
                break

        proc.wait()
        script_path.unlink(missing_ok=True)

        if timed_out:
            print_status("Kokoro TTS download timed out (>15 min).  "
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
    """Download the selected embedding model to local cache."""
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                     ("python.exe" if PLATFORM == "windows" else "python"))
    cache_dir = str(BASE_DIR / "data" / "embedding_cache")
    cache_parent = str(BASE_DIR / "data")

    script = f'''
import os, sys
os.environ["TRANSFORMERS_CACHE"] = r"{cache_dir}"
os.environ["HF_HOME"] = r"{cache_parent}"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"{cache_dir}"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
print("[EMBED] Downloading and initializing embedding model: {model_name}")
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    
    # Initialize model (downloads weights, config, pooling)
    print("[EMBED] Loading SentenceTransformer model...")
    model = SentenceTransformer("{model_name}", cache_folder=r"{cache_dir}")
    
    # Explicitly download and cache tokenizer files (vocab, merges, etc.)
    print("[EMBED] Downloading and caching tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained("{model_name}", cache_folder=r"{cache_dir}")
    
    # Run a test encode to ensure all runtime files are loaded/cached
    print("[EMBED] Running test encoding to finalize cache...")
    test = model.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
    
    print(f"[EMBED] Model and tokenizer fully initialized and cached (dim={{test.shape[1]}})")
    sys.exit(0)
except Exception as e:
    print(f"[EMBED] Error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    script_path = TEMP_DIR / "download_embedding.py"
    try:
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

        print_status(f"Downloading embedding model: {model_name}...")
        result = subprocess.run(
            [python_exe, str(script_path)],
            capture_output=True, text=True, timeout=600
        )

        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"  {line}")

        script_path.unlink(missing_ok=True)

        if result.returncode == 0:
            print_status(f"Embedding model installed: {model_name}")
            return True
        else:
            print_status("Embedding model download failed", False)
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
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
        "filter_rules": [["\r\n", "\n"], ["\r", "\n"]],
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

        platform_val      = system.get('platform', PLATFORM)
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
        platform=platform_val,
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

def run_installer():
    """Main installer flow with new menu structure."""

    if not check_version_compatibility():
        print("\nSystem requirements not met. Installation cannot continue.")
        return

    # Run detections early (needed for main menu display)
    run_detections_once()
    if PLATFORM == "windows":
        detect_windows_version()
    else:
        detect_linux_version()

    # ── Menu 1: Main Menu ──────────────────────────────────────────────
    main_choice = show_main_menu()
    if main_choice == 'A':
        print("\nInstallation abandoned.")
        return

    if main_choice == '3':
        refresh_configs()
        return

    is_clean_install = (main_choice == '1')

    # ── Menu 2: Embedding Model ────────────────────────────────────────
    embed_choice = show_embedding_menu()
    if embed_choice == 'A':
        print("\nInstallation abandoned.")
        return
    embedding_model = EMBEDDING_MODELS[embed_choice]["name"]

    # ── Menu 3: Backend ────────────────────────────────────────────────
    backend_choice = show_backend_menu()
    if backend_choice == 'A':
        print("\nInstallation abandoned.")
        return
    backend_keys = list(BACKEND_OPTIONS.keys())
    backend = backend_keys[int(backend_choice) - 1]

    # ── Menu 4: TTS Voice Pack ─────────────────────────────────────────
    tts_choice = show_tts_menu()
    if tts_choice == 'A':
        print("\nInstallation abandoned.")
        return
    voice_pack = KOKORO_VOICE_PACKS[tts_choice]

    # ── Execute Installation ───────────────────────────────────────────
    clear_screen()
    print_header("Installing...")
    if is_clean_install:
        print_status("Starting Clean Install...")
        if VENV_DIR.exists():
            _force_rmtree(VENV_DIR)
            print_status("Removed existing virtual environment")
    else:
        print_status("Starting Check/Install...")

    # Create directories
    create_files_and_directories(backend)

    # Create/ensure venv
    if is_clean_install:
        if not create_venv():
            return
    else:
        if not ensure_venv():
             return

    # Install system dependencies (Linux only) - NOW FATAL
    if PLATFORM == "linux":
        if not install_linux_system_dependencies(backend):
            print_status("Linux system dependencies installation failed. Installation aborted.", False)
            return

    # Install Python dependencies (includes llama-cpp-python wheel) - NOW FATAL
    if not install_python_deps(backend):
        print_status("Python dependency installation failed. Installation aborted.", False)
        return

    # Install optional file format support - NOW FATAL
    if not install_optional_file_support():
        print_status("Optional file support installation failed. Installation aborted.", False)
        return

    # Download embedding model to cache - NOW FATAL
    if not download_embedding_model(embedding_model):
         print_status("Embedding model download failed. Installation aborted.", False)
         return

    # Download Kokoro TTS model + voices for selected pack - NOW FATAL
    if not download_kokoro_voices(tts_choice):
        print_status("Kokoro TTS download failed. Installation aborted.", False)
        return

    # ── Create configuration files (Only reached if ALL above steps succeeded) ─
    info = BACKEND_OPTIONS[backend]
    backend_type = _determine_backend_type(backend)
    info = BACKEND_OPTIONS[backend]
    backend_type = _determine_backend_type(backend)

    create_system_ini(
        platform=PLATFORM,
        os_version=f"Windows {WINDOWS_VERSION}" if PLATFORM == "windows" and WINDOWS_VERSION else (OS_VERSION or "unknown"),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        backend_type=backend_type,
        embedding_model=embedding_model,
        vulkan_available=_DETECTED_VULKAN,
        llama_cli_path=info.get("cli_path"),
        llama_bin_path=info.get("dest"),
        tts_engine="kokoro",
        tts_pack=int(tts_choice),
        tts_default_voice_id=voice_pack["default_voice_id"],
        tts_default_voice_name=voice_pack["default_voice_name"],
        tts_enabled_voices=voice_pack["voice_ids"],
        browser_acceleration=_DETECTED_DX_CAPABLE,
        dx_feature_level=_DETECTED_DX_LEVEL,
    )

    # Patch wheel version into INI (written by install_python_deps)
    if _INSTALLED_LLAMA_WHEEL_VERSION:
        update_ini_wheel_version(_INSTALLED_LLAMA_WHEEL_VERSION)

    # Create configuration.json and preferences.json (existing values preserved)
    create_config_jsons()

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