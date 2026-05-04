"""Diagnose Azure OpenAI latency: HTTP-level logging + a couple of back-to-back calls."""

import logging
import os
import time

from openai import AzureOpenAI

# Minimal .env loader (avoid extra deps)
def _load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_env()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-12-01-preview",
    max_retries=0,
)

PROMPT = "Summarize: a small bug where the login button is misaligned on Safari."

for i in range(3):
    print(f"\n=== call {i+1} ===", flush=True)
    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.2,
            max_tokens=200,
        )
        elapsed = time.perf_counter() - start
        usage = resp.usage
        print(f"OK in {elapsed:.2f}s  prompt={usage.prompt_tokens} completion={usage.completion_tokens}", flush=True)
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"ERR in {elapsed:.2f}s: {type(e).__name__}: {e}", flush=True)
        # Print response headers if available
        if hasattr(e, "response") and e.response is not None:
            print("Response headers:", dict(e.response.headers), flush=True)
            try:
                print("Response body:", e.response.text, flush=True)
            except Exception:
                pass
