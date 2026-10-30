"""Smoke-test the two already-running Crux vLLM servers.

Start the configured Qwen and Llama vLLM servers first, then run:
    python scripts/test_model_servers.py

Defaults:
    Qwen  : http://127.0.0.1:8000, model name qwen3-8b
    Llama : http://127.0.0.1:8001, model name llama-3.1-8b-instruct

Override the base URLs with VLLM_BASE_URL_A and VLLM_BASE_URL_B (as in .env, e.g. http://127.0.0.1:8000/v1).

This sends plain requests without our backend's per-request settings (Qwen's enable_thinking: false), so a
<think> section here is expected. To check the real pipeline path, use `crux smoke consultants`.
Lives in scripts/, not tests/: it needs live servers.
"""
#test_model_servers.py
import os
import requests
import sys


def _server_url(env_name, default):
    """Server root from the environment; .env values end in /v1, which this script adds itself."""
    url = (os.environ.get(env_name) or default).rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


SERVERS = [
    {
        "name": "Qwen",
        "url": _server_url("VLLM_BASE_URL_A", "http://127.0.0.1:8000"),
        "model": "qwen3-8b",
        "prompt": "Reply with exactly: QWEN OK"
    },
    {
        "name": "Llama",
        "url": _server_url("VLLM_BASE_URL_B", "http://127.0.0.1:8001"),
        "model": "llama-3.1-8b-instruct",
        "prompt": "Reply with exactly: LLAMA OK"
    }
]


def check_health(server):
    try:
        response = requests.get(
            server["url"] + "/health",
            timeout=10
        )

        if response.status_code == 200:
            print("[PASS]", server["name"], "health: HTTP 200")
            return True

        print("[FAIL]", server["name"], "health:", response.status_code)

    except requests.RequestException as error:
        print("[FAIL]", server["name"], "health:", error)

    return False


def test_generation(server):
    try:
        response = requests.post(
            server["url"] + "/v1/chat/completions",
            json={
                "model": server["model"],
                "messages": [
                    {
                        "role": "user",
                        "content": server["prompt"]
                    }
                ],
                "max_tokens": 40,
                "temperature": 0
            },
            timeout=120
        )

        print("\n", server["name"], "HTTP:", response.status_code)

        if response.status_code != 200:
            print(response.text[:2000])
            return False

        result = response.json()

        generated = result["choices"][0]["message"]["content"]

        print("Generated response:")
        print(generated)

        if server["name"] == "Qwen" and "<think>" in generated:
            print("[NOTE] Qwen includes a think section.")

        return bool(generated.strip())

    except requests.RequestException as error:
        print("[FAIL]", server["name"], error)
        return False

    except (ValueError, KeyError, IndexError) as error:
        print("[FAIL] Invalid response:", error)
        return False


def main():
    print("CRUX AUDITED CONSENSUS")
    print("SIMULTANEOUS MODEL API TEST")
    print("=" * 45)

    health_results = {}

    for server in SERVERS:
        health_results[server["name"]] = check_health(server)

    if not all(health_results.values()):
        print("\nAt least one server is unavailable.")
        print("Make sure both vLLM servers are running.")
        return 1

    print("\nBoth servers are healthy!")
    print("\nTesting generation...")

    results = {}

    for server in SERVERS:
        results[server["name"]] = test_generation(server)

    print("\n" + "=" * 45)
    print("FINAL RESULTS")

    for name, passed in results.items():
        print(name, ":", "PASS" if passed else "FAIL")

    if all(results.values()):
        print("\nBoth model APIs passed!")
        return 0

    print("\nOne or more tests failed.")
    return 1

if __name__ == "__main__":
    sys.exit(main())