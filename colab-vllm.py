# -*- coding: utf-8 -*-
"""Host vLLM on Colab and expose via ngrok for the Agentic Search search_agent mode.

Run this file top-to-bottom in a Colab notebook cell or as a script.

Prerequisites:
  - Colab runtime with GPU (T4 free tier works; A100 is faster)
  - Free ngrok account — copy your authtoken from https://dashboard.ngrok.com/
"""

import ctypes
import glob
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PORT = 8080
NGROK_AUTHTOKEN = "YOUR_NGROK_AUTHTOKEN"  # ← paste your token here

# ── Section 1: Check GPU ──────────────────────────────────────────────────────

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError("No GPU — go to Runtime > Change runtime type and select a GPU.")
print(result.stdout)

# ── Section 2: Install missing dependencies ───────────────────────────────────
# Colab runtimes already include vLLM (0.20+) and transformers (5.x).
# Only pyngrok and nest_asyncio need to be added.
# Colab cell equivalent:
#   !pip install pyngrok nest_asyncio -q

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pyngrok", "nest_asyncio"],
    check=True,
)

# ── Section 3: Kill any existing vLLM process ────────────────────────────────
# pkill targets only the vLLM process by name — does not kill the kernel.

subprocess.run(["pkill", "-f", "vllm"], capture_output=True)
time.sleep(2)
print(f"Any previous vLLM process stopped. Port {PORT} is free.")

# ── Section 3.5: Preload libcudart so vllm._C can link against it ─────────────
# pip-installed CUDA packages put their .so files under site-packages/nvidia/*/lib.
# dlopen() won't find them unless we either set LD_LIBRARY_PATH or preload the
# library with RTLD_GLOBAL — the latter makes symbols globally visible to all
# subsequently loaded extensions (including vllm._C).


def _resolve_libcudart() -> str | None:
    """Return a path to libcudart.so.13 (real or compat symlink), or None."""
    # 1. Already present as .so.13
    for _pattern in [
        "/usr/local/lib/python*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.13*",
        "/usr/local/lib/python*/dist-packages/nvidia/*/lib/libcudart.so.13*",
        "/usr/local/cuda*/targets/*/lib/libcudart.so.13*",
        "/usr/local/cuda*/lib64/libcudart.so.13*",
        "/usr/lib/x86_64-linux-gnu/libcudart.so.13*",
    ]:
        _hits = sorted(glob.glob(_pattern))
        if _hits:
            return _hits[0]

    _ldc = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
    for _line in _ldc.stdout.splitlines():
        if "libcudart.so.13" in _line and "=>" in _line:
            return _line.split("=>")[-1].strip()

    # 2. CUDA 13 not found — create a compat symlink from libcudart.so.12.
    # CUDA 12 and 13 share the same core ABI for inference workloads.
    for _cudart12 in [
        "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
    ]:
        if os.path.exists(_cudart12):
            _compat = "/tmp/libcudart.so.13"
            subprocess.run(["ln", "-sf", _cudart12, _compat], check=True)
            print(f"Created CUDA 12→13 compat symlink: {_compat} -> {_cudart12}")
            return _compat

    return None


_cudart_path = _resolve_libcudart()
if _cudart_path:
    _lib_dir = os.path.dirname(_cudart_path)
    _existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = _lib_dir + (
        ":" + _existing_ld if _existing_ld else ""
    )
    try:
        ctypes.CDLL(_cudart_path, mode=ctypes.RTLD_GLOBAL)
        print(f"Preloaded {_cudart_path}")
    except OSError as _e:
        print(f"Warning: found {_cudart_path} but could not preload it: {_e}")
else:
    print("Warning: libcudart.so.13 not found — vllm may fail to import vllm._C")

# ── Section 4: Start vLLM server in-process (~60s on A100) ───────────────────
# Running in the same Python process avoids subprocess linker path issues.
# Logs print to stdout as normal. libcudart is already loaded above.


def _run_vllm() -> None:
    _old_argv = sys.argv[:]
    sys.argv = [
        "vllm",
        "serve",
        MODEL,
        "--port",
        str(PORT),
        "--dtype",
        "float16",
        "--max-model-len",
        "4096",
    ]
    try:
        from vllm.entrypoints.cli.main import main  # noqa: PLC0415

        main()
    except SystemExit:
        pass
    finally:
        sys.argv = _old_argv


threading.Thread(target=_run_vllm, daemon=True).start()

print(f"Waiting for vLLM to load {MODEL} (up to 3 min)...")
for i in range(36):
    time.sleep(5)
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
        print(f"\nvLLM ready after ~{(i + 1) * 5}s.")
        break
    except Exception:
        print(".", end="", flush=True)
else:
    raise RuntimeError("vLLM did not start in 3 min — check logs above for errors.")

# ── Section 5: Expose via ngrok ───────────────────────────────────────────────

import nest_asyncio  # noqa: E402
from pyngrok import ngrok  # noqa: E402

nest_asyncio.apply()

ngrok.set_auth_token(NGROK_AUTHTOKEN)
tunnel = ngrok.connect(PORT, "http")
public_url = tunnel.public_url
print(f"\nvLLM public URL: {public_url}")
print("\nAdd to your local .env:")
print(f"  SEARCH_AGENT_MODEL={MODEL}")
print(f"  SEARCH_AGENT_VLLM_URL={public_url}")

# ── Section 6: Smoke test ────────────────────────────────────────────────────

req = urllib.request.Request(
    f"http://localhost:{PORT}/v1/completions",
    data=json.dumps(
        {
            "model": MODEL,
            "prompt": "What is FAISS?",
            "max_tokens": 64,
            "temperature": 0,
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)
resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
print("Smoke test:", resp["choices"][0]["text"].strip())
