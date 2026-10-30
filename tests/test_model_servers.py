"""Smoke-test the two already-running Crux vLLM servers.

Start the configured Qwen and Llama vLLM servers first, then run:
    python scripts/test_model_servers.py

Defaults:
    Qwen  : http://127.0.0.1:8000, model name qwen3-8b
    Llama : http://127.0.0.1:8001, model name llama-3.1-8b-instruct

Override the base URLs with VLLM_BASE_URL_A and VLLM_BASE_URL_B.
"""

import os
import sys

import requests

SERVERS = [
    {
        "name": "Qwen",
        "base_url": os.getenv("VLLM_BASE_URL_A", "http://127.0.0.1:8000"),
        "model": "qwen3-8b",
        "prompt": "Reply with exactly: QWEN OK",
    },
    {
        "name": "Llama",
        "base_url": os.getenv("VLLM_BASE_URL_B", "http://127.0.0.1:8001"),
        "model": "llama-3.1-8b-instruct",
        "prompt": "Reply with exactly: LLAMA OK",
    },
]


def check_health(server):
    """Check whether the vLLM server is healthy."""
    url = f"{server['base_url'].rstrip('/')}/health"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"[PASS] {server['name']} health: HTTP 200")
            return True
        print(f"[FAIL] {server['name']} health: HTTP {response.status_code}")
    except requests.RequestException as exc:
        print(f"[FAIL] {server['name']} health: {exc}")
    return False


def test_generation(server):
    """Send one chat-completion request to a healthy server."""
    url = f"{server['base_url'].rstrip('/')}/v1/chat/completions"
    payload = {
        "model": server["model"],
        "messages": [{"role": "user", "content": server["prompt"]}],
        "max_tokens": 40,
        "temperature": 0,
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"\n{server['name']} HTTP: {response.status_code}")

        if response.status_code != 200:
            print(response.text[:2000])
            return False

        body = response.json()
        generated = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not generated:
            print(f"[FAIL] {server['name']} returned empty content")
            return False

        print(f"{server['name']} generated:\n{generated.strip()}")

        if server["name"] == "Qwen" and "<think>" in generated:
            print(
                "[NOTE] Qwen output contains '<think>'. "
                "Verify the configured chat-template behavior separately."
            )
        return True

    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        print(f"[FAIL] {server['name']} generation: {exc}")
        return False


def main():
    print("Crux Audited Consensus - simultaneous model API smoke test")
    print("=" * 60)

    health = {server["name"]: check_health(server) for server in SERVERS}
    if not all(health.values()):
        print("\nAt least one server is unhealthy. Start both servers and retry.")
        return 1

    print("\nBoth servers are healthy. Testing generation...")
    results = {
        server["name"]: test_generation(server)
        for server in SERVERS
    }

    print("\nSUMMARY")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if all(results.values()):
        print("\nBoth model APIs passed the simultaneous smoke test.")
        return 0

    print("\nOne or more model API tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
