"""In-process Hugging Face transformers backend for the judges, with the TRUE/FALSE next-token readout.

torch and transformers are imported lazily, so the readout helpers below are importable (and tested) without them.

c_prob readout (audit 4.2): generate the verdict at T=0, cut the completion right after the last ``VERDICT:``,
run one forward pass on chat-template prompt + that prefix, and sum the next-token probabilities of the first
tokens of every TRUE / FALSE spelling. ``p_mass_tf`` (raw P(TRUE) + P(FALSE)) is logged so a format failure
is not mistaken for a confident verdict.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from crux.backends.base import Backend, BackendError, ModelSpec, Registry

TRUE_VARIANTS = (" TRUE", "TRUE", " True", "True", " true", "true")
FALSE_VARIANTS = (" FALSE", "FALSE", " False", "False", " false", "false")


class VerdictMarkerMissing(BackendError):
    pass


def cut_after_marker(completion: str, marker: str = "VERDICT:") -> str:
    """Prefix of ``completion`` ending right after the last ``marker``."""
    position = completion.rfind(marker)
    if position < 0:
        raise VerdictMarkerMissing(f"marker {marker!r} not found in completion")
    return completion[: position + len(marker)]


def first_token_ids(tokenizer: Any, variants: Iterable[str]) -> List[int]:
    ids = set()
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if encoded:
            ids.add(int(encoded[0]))
    return sorted(ids)


def verdict_token_ids(tokenizer: Any) -> Tuple[List[int], List[int]]:
    """First-token ids for TRUE and FALSE spellings. Raises if the two sets overlap or either is empty."""
    true_ids = first_token_ids(tokenizer, TRUE_VARIANTS)
    false_ids = first_token_ids(tokenizer, FALSE_VARIANTS)
    if not true_ids or not false_ids:
        raise BackendError("tokenizer produced no first tokens for TRUE or FALSE")
    overlap = set(true_ids) & set(false_ids)
    if overlap:
        raise BackendError(f"TRUE and FALSE share first tokens {sorted(overlap)}; c_prob would be meaningless")
    return true_ids, false_ids


def verdict_readout(prob_of: Callable[[int], float], true_ids: Sequence[int],
                    false_ids: Sequence[int]) -> Dict[str, Any]:
    """Sum next-token probabilities over each id set and renormalize over the two."""
    p_true = float(sum(prob_of(i) for i in true_ids))
    p_false = float(sum(prob_of(i) for i in false_ids))
    mass = p_true + p_false
    return {
        "p_true_raw": p_true,
        "p_false_raw": p_false,
        "p_mass_tf": mass,
        "p_true": p_true / mass if mass > 0 else None,
        "true_token_ids": list(true_ids),
        "false_token_ids": list(false_ids),
    }


class HFLocalBackend(Backend):
    name = "hf_local"

    def __init__(self, registry: Registry, *, device_map: str = "auto"):
        super().__init__(registry)
        self.device_map = device_map
        self._loaded: Dict[str, Tuple[Any, Any]] = {}

    def _load(self, spec: ModelSpec) -> Tuple[Any, Any]:
        if spec.key not in self._loaded:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if spec.quantization != "none":
                raise BackendError(f"{spec.key}: hf_local supports quantization 'none' only")
            tokenizer = AutoTokenizer.from_pretrained(spec.hf_repo, revision=spec.revision)
            model = AutoModelForCausalLM.from_pretrained(
                spec.hf_repo, revision=spec.revision, torch_dtype=getattr(torch, spec.dtype),
                device_map=self.device_map,
            )
            model.eval()
            verdict_token_ids(tokenizer)  # fail at load time, not mid-run, if TRUE/FALSE tokens collide
            self._loaded[spec.key] = (tokenizer, model)
        return self._loaded[spec.key]

    def _versions(self, model: Any) -> Dict[str, Any]:
        import torch
        import transformers

        return {"transformers_version": transformers.__version__, "torch_version": torch.__version__,
                "device": str(model.device)}

    @staticmethod
    def _render(spec: ModelSpec, tokenizer: Any, messages: List[Dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             **spec.chat_template_kwargs)

    def _generate(self, spec: ModelSpec, messages: List[Dict[str, str]], *, temperature: float,
                  top_p: Optional[float], max_tokens: int, seed: Optional[int],
                  n: int) -> Tuple[List[str], Dict[str, Any]]:
        import torch
        from transformers import set_seed

        tokenizer, model = self._load(spec)
        prompt = self._render(spec, tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        if seed is not None:
            set_seed(seed)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        kwargs: Dict[str, Any] = {"max_new_tokens": max_tokens, "num_return_sequences": n, "pad_token_id": pad_id}
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
            if top_p is not None:
                kwargs["top_p"] = top_p
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            output = model.generate(**inputs, **kwargs)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        eos_ids = model.generation_config.eos_token_id
        eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids]) - {None}
        # Same vocabulary as the OpenAI API: "stop" if an end-of-sequence token was produced, else "length".
        finish_reasons = ["stop" if eos_ids & set(row.tolist()) else "length" for row in new_tokens]
        return texts, {**self._versions(model), "finish_reasons": finish_reasons}

    def verdict_probs(self, model_key: str, messages: List[Dict[str, str]], completion: str,
                      *, marker: str = "VERDICT:") -> Dict[str, Any]:
        import torch

        spec = self.spec(model_key)
        tokenizer, model = self._load(spec)
        text = self._render(spec, tokenizer, messages) + cut_after_marker(completion, marker)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        true_ids, false_ids = verdict_token_ids(tokenizer)
        readout = verdict_readout(lambda token_id: probs[token_id].item(), true_ids, false_ids)
        return {**readout, **spec.describe(), **self._versions(model)}
