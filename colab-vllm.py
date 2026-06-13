# -*- coding: utf-8 -*-
"""Host vLLM on Colab and expose via ngrok for the Agentic Search search_agent mode.

Run this file top-to-bottom in a Colab notebook cell or as a script.

Prerequisites:
  - Colab runtime with GPU (T4 free tier works; A100 is faster)
  - Free ngrok account — copy your authtoken from https://dashboard.ngrok.com/
"""

import json
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

# ── Section 4: Start vLLM server (~60s on A100) ───────────────────────────────
# Logs stream below. Wait for "Application startup complete" before Section 5.

proc = subprocess.Popen(
    [
        "vllm",
        "serve",
        MODEL,
        "--port",
        str(PORT),
        "--dtype",
        "float16",
        "--max-model-len",
        "4096",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)


def _stream_logs() -> None:
    for line in proc.stdout:
        print(line, end="", flush=True)


threading.Thread(target=_stream_logs, daemon=True).start()

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
