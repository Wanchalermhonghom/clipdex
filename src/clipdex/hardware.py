"""Detect the machine and pick a model profile that fits its memory."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field

GB = 1_000_000_000


@dataclass
class Hardware:
    system: str  # Darwin | Linux | Windows
    machine: str  # arm64 | x86_64 | AMD64
    cpu: str
    ram_gb: float
    accelerator: str  # mps | cuda | cpu
    gpu_name: str = ""
    vram_gb: float = 0.0

    @property
    def apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine == "arm64"

    def summary(self) -> str:
        gpu = f", GPU {self.gpu_name} ({self.vram_gb:.0f} GB)" if self.accelerator == "cuda" else ""
        return (f"{self.cpu} · {self.ram_gb:.0f} GB RAM · {self.system} {self.machine} · "
                f"accelerator {self.accelerator}{gpu}")


@dataclass
class Profile:
    name: str
    description: str
    settings: dict
    ram_index_gb: float  # rough peak for 'clipdex index'
    ram_mcp_gb: float  # rough peak for 'clipdex mcp'
    notes: list[str] = field(default_factory=list)


# Memory numbers are estimates: model weights (fp16/bf16) plus about 30 % for activations.
# Weights: SigLIP 2 base 0.75 GB, so400m 2.3 GB, Qwen3-VL-Embedding 2B 4.3 GB, 8B 16.3 GB,
# Whisper large-v3-turbo 1.6 GB. Measure your own peak with 'clipdex index --limit 3'.
PROFILES: dict[str, Profile] = {
    "apple-small": Profile(
        "apple-small", "Apple Silicon with less than 16 GB memory",
        {"siglip_model": "google/siglip2-base-patch16-256", "siglip_batch": 16, "qwen_enabled": False},
        ram_index_gb=3.5, ram_mcp_gb=1.5,
        notes=["No scene (action) search and no semantic speech search. Exact speech and visual search work."]),
    "apple-medium": Profile(
        "apple-medium", "Apple Silicon with 16-47 GB memory",
        {"siglip_model": "google/siglip2-so400m-patch14-384", "qwen_enabled": True,
         "qwen_model": "Qwen/Qwen3-VL-Embedding-2B", "qwen_batch": 4},
        ram_index_gb=10.0, ram_mcp_gb=8.0,
        notes=["While the indexer runs on a 16-24 GB Mac, start the MCP server with --no-qwen or close other apps."]),
    "apple-large": Profile(
        "apple-large", "Apple Silicon with 48 GB memory or more",
        {"siglip_model": "google/siglip2-so400m-patch14-384", "qwen_enabled": True,
         "qwen_model": "Qwen/Qwen3-VL-Embedding-8B", "qwen_batch": 4},
        ram_index_gb=24.0, ram_mcp_gb=21.0,
        notes=["The search thresholds were calibrated with the 2B model. Check them with --explain."]),
    "nvidia-small": Profile(
        "nvidia-small", "NVIDIA GPU with less than 10 GB VRAM",
        {"siglip_model": "google/siglip2-so400m-patch14-384", "qwen_enabled": False},
        ram_index_gb=4.5, ram_mcp_gb=3.0),
    "nvidia-medium": Profile(
        "nvidia-medium", "NVIDIA GPU with 10-23 GB VRAM",
        {"siglip_model": "google/siglip2-so400m-patch14-384", "qwen_enabled": True,
         "qwen_model": "Qwen/Qwen3-VL-Embedding-2B", "qwen_batch": 8},
        ram_index_gb=10.0, ram_mcp_gb=8.0),
    "nvidia-large": Profile(
        "nvidia-large", "NVIDIA GPU with 24 GB VRAM or more",
        {"siglip_model": "google/siglip2-so400m-patch14-384", "qwen_enabled": True,
         "qwen_model": "Qwen/Qwen3-VL-Embedding-8B", "qwen_batch": 8},
        ram_index_gb=24.0, ram_mcp_gb=21.0,
        notes=["The search thresholds were calibrated with the 2B model. Check them with --explain."]),
    "cpu": Profile(
        "cpu", "No supported GPU: small and slow, but it works",
        {"siglip_model": "google/siglip2-base-patch16-256", "siglip_batch": 8, "qwen_enabled": False,
         "asr_model": "small"},
        ram_index_gb=3.0, ram_mcp_gb=1.5,
        notes=["Expect indexing slower than real time. Whisper 'small' is less accurate than large-v3-turbo."]),
}


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _ram_bytes() -> int:
    if sys.platform == "darwin":
        value = _sysctl("hw.memsize")
        return int(value) if value.isdigit() else 0
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            return 0
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    return 0


def _cpu_name() -> str:
    if sys.platform == "darwin":
        return _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown CPU"
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown CPU"


def detect() -> Hardware:
    hw = Hardware(system=platform.system(), machine=platform.machine(), cpu=_cpu_name(),
                  ram_gb=_ram_bytes() / GB, accelerator="cpu")
    try:
        import torch

        if torch.backends.mps.is_available():
            hw.accelerator = "mps"
        elif torch.cuda.is_available():
            hw.accelerator = "cuda"
            props = torch.cuda.get_device_properties(0)
            hw.gpu_name = props.name
            hw.vram_gb = props.total_memory / GB
    except Exception:  # noqa: BLE001 - torch missing or broken: treat as CPU
        pass
    return hw


def choose_profile(hw: Hardware) -> Profile:
    if hw.accelerator == "mps":
        if hw.ram_gb < 15.5:  # "16 GB" Macs report a little less than 16e9 bytes
            return PROFILES["apple-small"]
        if hw.ram_gb < 47.5:
            return PROFILES["apple-medium"]
        return PROFILES["apple-large"]
    if hw.accelerator == "cuda":
        if hw.vram_gb < 9.5:
            return PROFILES["nvidia-small"]
        if hw.vram_gb < 23.5:
            return PROFILES["nvidia-medium"]
        return PROFILES["nvidia-large"]
    return PROFILES["cpu"]


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(type(value))


def render_config(profile: Profile, hw: Hardware | None = None) -> str:
    lines = [
        f"# clipdex config — profile '{profile.name}': {profile.description}",
    ]
    if hw is not None:
        lines.append(f"# Detected: {hw.summary()}")
    lines += [
        f"# Estimated peak memory: indexer about {profile.ram_index_gb:.0f} GB, "
        f"MCP server about {profile.ram_mcp_gb:.0f} GB.",
        "# All settings and their defaults: clipdex.example.toml in the repository.",
        "",
    ]
    lines += [f"{k} = {_toml_value(v)}" for k, v in profile.settings.items()]
    return "\n".join(lines) + "\n"


def memory_report() -> str:
    """Peak memory of this process, for tuning. Apple GPU memory is reported separately."""
    parts: list[str] = []
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        parts.append(f"process peak {peak_bytes / GB:.1f} GB")
    except (ImportError, OSError):
        pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                parts.append(f"torch MPS now {torch.mps.driver_allocated_memory() / GB:.1f} GB")
            elif torch.cuda.is_available():
                parts.append(f"torch CUDA peak {torch.cuda.max_memory_allocated() / GB:.1f} GB")
        except Exception:  # noqa: BLE001
            pass
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        for getter in (getattr(mx, "get_peak_memory", None), getattr(getattr(mx, "metal", None), "get_peak_memory", None)):
            if callable(getter):
                try:
                    parts.append(f"MLX peak {getter() / GB:.1f} GB")
                    break
                except Exception:  # noqa: BLE001
                    continue
    return ", ".join(parts) if parts else "not available"
