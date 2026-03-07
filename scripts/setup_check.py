#!/usr/bin/env python3
"""
setup_check.py — Hardware detection script for the local AI video series.

Detects your system hardware, checks MLX availability, and recommends
the best models to run locally based on your available RAM.

Usage:
    python setup_check.py

Dependencies:
    pip install psutil        # required
    pip install mlx-lm        # optional, Apple Silicon only
"""

import platform
import subprocess
import sys
import shutil

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def _c(text, *codes):
    """Wrap text in ANSI colour codes (skipped on Windows cmd without colour support)."""
    prefix = "".join(codes)
    return f"{prefix}{text}{RESET}"

def header(title):
    width = 60
    bar = "─" * width
    print()
    print(_c(f"┌{bar}┐", CYAN, BOLD))
    pad = (width - len(title)) // 2
    print(_c(f"│{' ' * pad}{title}{' ' * (width - pad - len(title))}│", CYAN, BOLD))
    print(_c(f"└{bar}┘", CYAN, BOLD))

def section(title):
    print()
    print(_c(f"  ▸ {title}", BOLD, YELLOW))
    print(_c("  " + "─" * 56, DIM))

def row(label, value, indent=4):
    label_str = _c(f"{label:<28}", DIM)
    print(f"{' ' * indent}{label_str} {value}")

def check(label, ok, detail="", indent=4):
    icon  = _c("✓", GREEN) if ok else _c("✗", RED)
    label_str = f"{label:<40}"
    detail_str = _c(f"  {detail}", DIM) if detail else ""
    print(f"{' ' * indent}{icon}  {label_str}{detail_str}")

# ---------------------------------------------------------------------------
# 1. OS Detection
# ---------------------------------------------------------------------------

def detect_os():
    system = platform.system()   # 'Darwin', 'Linux', 'Windows'
    release = platform.release()
    version = platform.version()
    machine = platform.machine() # 'arm64', 'x86_64', etc.
    return {
        "system": system,
        "release": release,
        "version": version,
        "machine": machine,
        "is_mac": system == "Darwin",
        "is_linux": system == "Linux",
        "is_windows": system == "Windows",
    }

# ---------------------------------------------------------------------------
# 2. Chip / GPU Detection
# ---------------------------------------------------------------------------

def _run(cmd, timeout=8):
    """Run a shell command; return (stdout, returncode)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip(), result.returncode
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "", 1

def detect_apple_silicon(os_info):
    """Return chip brand string on macOS, else None."""
    if not os_info["is_mac"]:
        return None
    out, rc = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if rc == 0 and out:
        return out
    # macOS 26+ removed machdep.cpu.brand_string — use system_profiler
    out2, rc2 = _run(["/usr/sbin/system_profiler", "SPHardwareDataType"], timeout=10)
    if rc2 == 0 and out2:
        for line in out2.splitlines():
            line = line.strip()
            if "Chip" in line and ":" in line:
                return line.split(":", 1)[1].strip()
    # Final fallback
    out3, rc3 = _run(["sysctl", "-n", "hw.model"])
    return out3 if rc3 == 0 else "Apple Silicon (unknown variant)"

def detect_nvidia_gpu():
    """
    Query nvidia-smi for GPU name and VRAM.
    Returns list of dicts: [{name, vram_mib}] or [].
    """
    if not shutil.which("nvidia-smi"):
        return []
    out, rc = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0 or not out:
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            name = parts[0]
            try:
                vram_mib = int(parts[1])
            except ValueError:
                vram_mib = 0
            gpus.append({"name": name, "vram_mib": vram_mib})
    return gpus

# ---------------------------------------------------------------------------
# 3. RAM Detection (psutil)
# ---------------------------------------------------------------------------

def detect_ram():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_bytes": vm.total,
            "available_bytes": vm.available,
            "total_gib": vm.total / (1024 ** 3),
            "available_gib": vm.available / (1024 ** 3),
            "psutil_ok": True,
        }
    except ImportError:
        return {"psutil_ok": False}

# ---------------------------------------------------------------------------
# 4. MLX Detection
# ---------------------------------------------------------------------------

def detect_mlx():
    info = {"available": False, "version": None, "device": None}
    try:
        import mlx.core as mx          # type: ignore
        info["available"] = True
        try:
            import mlx                 # type: ignore
            info["version"] = getattr(mlx, "__version__", "unknown")
        except Exception:
            info["version"] = "unknown"
        try:
            info["device"] = str(mx.default_device())
        except Exception:
            info["device"] = "unknown"
    except ImportError:
        pass
    return info

# ---------------------------------------------------------------------------
# 5. Model Recommendations
# ---------------------------------------------------------------------------

MODELS = [
    {
        "name":    "Qwen3.5-35B-A3B (MoE)",
        "model_id": "RepublicOfKorokke/Qwen3.5-35B-A3B-mlx-lm-nvfp4",
        "size_gb": 20,
        "min_ram": 24,
        "note":    "~20 GB — excellent MoE, fast",
        "port":    8800,
    },
    {
        "name":    "Qwen3.5-9B",
        "model_id": "RepublicOfKorokke/Qwen3.5-9B-mlx-lm-mxfp4",
        "size_gb": 5,
        "min_ram": 8,
        "note":    "~5 GB — great mid-range sweet spot",
        "port":    8801,
    },
    {
        "name":    "Qwen3.5-2B",
        "model_id": "RepublicOfKorokke/Qwen3.5-2B-mlx-lm-nvfp4",
        "size_gb": 1.2,
        "min_ram": 4,
        "note":    "~1.2 GB — tiny, runs anywhere",
        "port":    8802,
    },
]

def tier_label(total_gib):
    if total_gib >= 128:
        return "128 GB+"
    elif total_gib >= 64:
        return "64 GB"
    elif total_gib >= 32:
        return "32 GB"
    else:
        return "≤ 16 GB"

def model_fits(model, total_gib):
    """True if the model's minimum RAM requirement is met with a small headroom buffer."""
    # Headroom: keep ~15 % free for OS / KV cache
    usable = total_gib * 0.85
    return usable >= model["size_gb"]

# ---------------------------------------------------------------------------
# 6. Quick-Start Commands
# ---------------------------------------------------------------------------

QUICKSTART_MAC = """\
  # 1. Install dependencies
  pip install psutil mlx-lm openai

  # 2. Launch MLX servers (one model per port for parallel inference)
  mlx_lm.server --model RepublicOfKorokke/Qwen3.5-35B-A3B-mlx-lm-nvfp4 --port 8800
  mlx_lm.server --model RepublicOfKorokke/Qwen3.5-9B-mlx-lm-mxfp4 --port 8801
  mlx_lm.server --model RepublicOfKorokke/Qwen3.5-2B-mlx-lm-nvfp4 --port 8802

  # 3. Test with curl
  curl http://localhost:8800/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
      "model": "RepublicOfKorokke/Qwen3.5-2B-mlx-lm-nvfp4",
      "messages": [{"role":"user","content":"Hello!"}]
    }'
"""

QUICKSTART_OTHER = """\
  # Install Ollama  →  https://ollama.com/download
  # Then pull a GGUF model (example):
  ollama pull qwen2.5:32b-instruct-q4_K_M

  # Run interactively:
  ollama run qwen2.5:32b-instruct-q4_K_M

  # Or serve via OpenAI-compatible API on port 11434:
  OLLAMA_HOST=0.0.0.0 ollama serve
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    header("Hardware Setup Check")

    # ── OS ──────────────────────────────────────────────────────────────────
    section("Operating System")
    os_info = detect_os()
    row("OS", f"{os_info['system']} {os_info['release']}")
    row("Architecture", os_info["machine"])
    row("Python", sys.version.split()[0])

    # ── Chip / GPU ───────────────────────────────────────────────────────────
    section("Chip / Accelerator")
    chip_label   = "unknown"
    nvidia_gpus  = []

    if os_info["is_mac"]:
        chip_str = detect_apple_silicon(os_info)
        chip_label = chip_str or "Apple Silicon"
        row("Chip", chip_label)
    else:
        nvidia_gpus = detect_nvidia_gpu()
        if nvidia_gpus:
            for i, gpu in enumerate(nvidia_gpus):
                vram_gib = gpu["vram_mib"] / 1024
                row(f"GPU {i}", f"{gpu['name']}  ({vram_gib:.1f} GiB VRAM)")
                chip_label = gpu["name"]
        else:
            row("GPU", _c("No NVIDIA GPU detected (nvidia-smi not found)", DIM))
            chip_label = "CPU only"

    # ── RAM ─────────────────────────────────────────────────────────────────
    section("Memory (RAM)")
    ram = detect_ram()
    if ram["psutil_ok"]:
        row("Total RAM",     f"{ram['total_gib']:.1f} GiB")
        row("Available RAM", f"{ram['available_gib']:.1f} GiB")
        total_gib = ram["total_gib"]
        avail_gib = ram["available_gib"]
    else:
        print(_c("    ✗  psutil not installed — run: pip install psutil", RED))
        total_gib = 0
        avail_gib = 0

    # ── MLX ─────────────────────────────────────────────────────────────────
    section("MLX Framework")
    mlx = detect_mlx()
    if mlx["available"]:
        check("mlx.core importable", True,
              f"version={mlx['version']}  device={mlx['device']}")
    else:
        check("mlx.core importable", False,
              "pip install mlx-lm  (Apple Silicon only)")

    # ── Model Recommendations ────────────────────────────────────────────────
    section("Model Recommendations")

    if not ram["psutil_ok"]:
        print(_c("    Cannot generate recommendations without RAM info.", RED))
    else:
        tier = tier_label(total_gib)
        print(f"    Detected tier: {_c(tier, BOLD, GREEN)}")
        print()

        if os_info["is_mac"]:
            print(f"    {'Model':<38} {'Fits?':<8} Note")
            print("    " + "─" * 56)
            for m in MODELS:
                fits = model_fits(m, total_gib)
                icon = _c("✓", GREEN) if fits else _c("✗", RED)
                name_str = _c(m["name"], BOLD) if fits else _c(m["name"], DIM)
                print(f"    {icon}  {name_str:<46} {_c(m['note'], DIM)}")
        else:
            # Non-Mac path: show NVIDIA VRAM-based guidance
            vram_gib = nvidia_gpus[0]["vram_mib"] / 1024 if nvidia_gpus else 0
            effective = max(vram_gib, total_gib)  # CPU offload possible
            print(f"    {'Model':<38} {'Fits?':<8} Note")
            print("    " + "─" * 56)
            for m in MODELS:
                fits = model_fits(m, effective)
                icon = _c("✓", GREEN) if fits else _c("✗", RED)
                name_str = _c(m["name"], BOLD) if fits else _c(m["name"], DIM)
                print(f"    {icon}  {name_str:<46} {_c(m['note'], DIM)}")
            print()
            print(_c("    ⓘ  Non-Mac detected — recommend Ollama + GGUF models:", YELLOW))
            print(_c("       https://ollama.com  |  https://huggingface.co/bartowski", DIM))

    # ── Quick Start ──────────────────────────────────────────────────────────
    section("Quick Start")
    if os_info["is_mac"] and mlx["available"]:
        print(_c("    Apple Silicon + MLX detected 🎉", GREEN, BOLD))
        print()
        print(QUICKSTART_MAC)
        fitting = [m for m in MODELS if model_fits(m, total_gib)]
        if fitting:
            total_gb = sum(m["size_gb"] for m in fitting)
            print(f"    Your {total_gib:.0f} GiB system can run {len(fitting)} model(s) simultaneously (~{total_gb:.0f} GB total)")
    elif os_info["is_mac"] and not mlx["available"]:
        print(_c("    Apple Silicon detected but MLX not installed:", YELLOW))
        print()
        print("    pip install mlx-lm")
        print()
        print(QUICKSTART_MAC)
    else:
        print(_c("    Linux / Windows — use Ollama + GGUF:", YELLOW))
        print()
        print(QUICKSTART_OTHER)

    # ── Footer ───────────────────────────────────────────────────────────────
    print()
    print(_c("  " + "─" * 58, DIM))
    print(_c("  github.com/shanemmattner/llm-lab-videos  ·  Happy hacking! 🚀", DIM))
    print()


if __name__ == "__main__":
    main()
