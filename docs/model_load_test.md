# Model Loading and Simultaneous Inference Test Report

**Project:** Crux – Audited Consensus  
**Test date:** 30 October 2026  
**Platform:** Kaggle Notebook  
**Overall result:** Both models loaded and served simultaneous API requests successfully.

## 1. Objective

Verify that the two configured consultant models can be loaded on separate NVIDIA GPUs and serve chat-completion requests at the same time using vLLM.

## 2. Test environment

| Component | Observed configuration |
|---|---|
| Platform | Kaggle Notebook |
| GPUs | 2 × NVIDIA Tesla T4 |
| Reported GPU memory | 15,360 MiB per GPU |
| Python | 3.12.3 |
| vLLM | 0.32.0 |
| Quantization | AWQ |
| Data type | FP16 (`half`) |
| Tensor parallel size | 1 |
| Maximum model length | 16,384 tokens |
| Maximum sequences | 1 |

## 3. Model configuration

| Setting | Consultant A | Consultant B |
|---|---|---|
| Model repository | `Qwen/Qwen3-8B-AWQ` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` |
| Revision | `4da05a8edb55c6046cce958586c33b61da07bb79` | `db1f81ad4b8c7e39777509fac66c652eb0a52f91` |
| GPU | GPU 0 | GPU 1 |
| Served model name | `qwen3-8b` | `llama-3.1-8b-instruct` |
| API port | `8000` | `8001` |

## 4. Environment setup

The following setup was completed in Kaggle:

1. Configured the notebook to use two Tesla T4 GPUs.
2. Cloned the project repository.
3. Created a Python 3.12.3 virtual environment.
4. Installed vLLM and its dependencies.
5. Verified the installed vLLM version and CUDA availability.

Environment verification output:

```text
vLLM: 0.32.0
CUDA: True
GPUs: 2
```

## 5. Individual model tests

Both models were first tested independently.

### 5.1 Qwen3-8B-AWQ

- GPU assignment: GPU 0
- API port: `8000`
- vLLM server reached application startup.
- Health endpoint returned HTTP 200.
- Chat-completion request returned HTTP 200.
- Generated output included a `<think>` section and did not follow the exact-response test prompt.

**Result:** Model initialization and API generation succeeded. Exact-output / thinking-disabled behavior remains to be configured and verified.

### 5.2 Llama 3.1-8B-Instruct-AWQ-INT4

- GPU assignment: GPU 1
- API port: `8001`
- vLLM server reached application startup.
- Health endpoint returned HTTP 200.
- Chat-completion request returned HTTP 200.
- Test prompt: `Reply with exactly: LLAMA OK`
- Generated response: `LLAMA OK`

**Result:** Passed individual loading and API generation test.

## 6. Simultaneous loading and inference test

Both vLLM servers were launched in the same Kaggle session:

- Qwen on GPU 0, port `8000`
- Llama on GPU 1, port `8001`

On the successful simultaneous run, both health endpoints returned HTTP 200:

```text
Qwen READY!
Llama READY!
FINAL STATUS: {'Qwen': True, 'Llama': True}
```

Both chat-completion requests were then sent while both servers were running:

```text
Qwen HTTP: 200
<think>
Okay, the user sent "Reply exactly: Qwen OK". They want a precise response

Llama HTTP: 200
Llama OK
```

**Result:** Both APIs successfully handled requests during simultaneous operation. Qwen generated text but did not follow the exact-output prompt because its response included a `<think>` section. Llama returned the expected exact response.

## 7. GPU memory observation

The `nvidia-smi` output captured after the simultaneous API requests showed:

| GPU | Model assignment | Memory used | Total memory |
|---|---|---:|---:|
| GPU 0 | Qwen | 13,349 MiB | 15,360 MiB |
| GPU 1 | Llama | 13,325 MiB | 15,360 MiB |

This confirms that both GPUs had model memory allocated at the time of the snapshot. GPU utilization was reported as 0% in that snapshot, which was taken after the generation requests had completed.

## 8. Issues and follow-up

### Qwen thinking output

Qwen returned a `<think>` section despite the intended configuration:

```yaml
chat_template_kwargs:
  enable_thinking: false
strip_think: true
```

The attempted vLLM CLI option for passing chat-template keyword arguments was not accepted by the installed vLLM version. The appropriate way to pass these settings for the project's installed vLLM version still needs to be confirmed.

This did not prevent model initialization or API connectivity, but it means the intended thinking-disabled output behavior has **not** been verified.

### Initial simultaneous launch attempt

An initial simultaneous launch attempt ended before the health checks completed. The servers were relaunched, and the subsequent simultaneous health checks and API requests passed. The successful second run is the basis for the result in this report.

## 9. Final test summary

| Test | Result |
|---|---|
| Python 3.12.3 environment | Passed |
| vLLM 0.32.0 installation | Passed |
| CUDA detection / two GPUs | Passed |
| Qwen individual initialization | Passed |
| Qwen individual API request | Passed |
| Llama individual initialization | Passed |
| Llama individual API request | Passed |
| Both models initialized simultaneously | Passed |
| Both health endpoints available simultaneously | Passed |
| Both chat-completion APIs returned HTTP 200 simultaneously | Passed |
| Qwen thinking-disabled behavior | Follow-up required |

## 10. Conclusion

Qwen3-8B-AWQ and Llama 3.1-8B-Instruct-AWQ-INT4 were successfully loaded on separate Tesla T4 GPUs in the same Kaggle session using vLLM 0.32.0.

Both servers became healthy simultaneously, and both chat-completion APIs returned HTTP 200 responses while both models were running. The GPU memory snapshot also showed approximately 13.3 GiB allocated on each GPU.

The core model-loading and simultaneous API smoke test passed. The remaining follow-up is to configure and verify Qwen's intended thinking-disabled chat-template behavior.
