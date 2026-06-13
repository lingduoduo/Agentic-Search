# -*- coding: utf-8 -*-
"""Host vLLM on Colab and expose via ngrok for the Agentic Search search_agent mode.

Run each section in order inside a Colab notebook.

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

# ── Section 1: Check GPU ──────────────────────────────────────────────────────

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if result.returncode != 0:
    print("No GPU — go to Runtime > Change runtime type and select a GPU.")
else:
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

# ── Section 3: Free port if already in use ───────────────────────────────────

subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)
print(f"Port {PORT} cleared.")

# ── Section 4: Start vLLM server (~60s on A100) ───────────────────────────────
# Watch logs for "Uvicorn running on" — that means it's ready.

proc = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
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
)


def _stream_logs() -> None:
    for line in proc.stdout:
        print(line.decode(), end="", flush=True)


threading.Thread(target=_stream_logs, daemon=True).start()

print(f"Waiting for vLLM to load {MODEL}...")
for i in range(36):  # up to 3 min
    time.sleep(5)
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2)
        print(f"\nvLLM ready after ~{(i + 1) * 5}s.")
        break
    except Exception:
        print(".", end="", flush=True)
else:
    print("\nTimed out — check the logs above for errors.")

# ── Section 5: Expose via ngrok ───────────────────────────────────────────────

import nest_asyncio  # noqa: E402
from pyngrok import ngrok  # noqa: E402

nest_asyncio.apply()

NGROK_AUTHTOKEN = "YOUR_NGROK_AUTHTOKEN"  # ← paste your token here
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
