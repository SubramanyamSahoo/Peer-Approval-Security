from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import os
import string
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .config import stable_int


class ContextOverflow(ValueError):
    """A recoverable missing observation for auxiliary monitors, never a truncation."""


@dataclass
class Job:
    key: str
    messages: list[dict]
    options: dict[str, str]
    order_key: str
    capture: bool = True
    patches: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class Scored:
    actions: list[str]
    probabilities: torch.Tensor
    logits: torch.Tensor
    hidden: torch.Tensor | None
    prompt_tokens: int
    prompt: str
    diagnostics: dict = field(default_factory=dict)

    def probability(self, action: str) -> torch.Tensor:
        if action not in self.actions:
            return self.probabilities.new_zeros(())
        return self.probabilities[self.actions.index(action)]

    def choose(self, key: str) -> str:
        # Stateless common random numbers keep draws unchanged after OOM re-batching.
        raw = stable_int(key)
        # Python integers above int64 overflow Torch's scalar arithmetic conversion.
        # Explicit float64 conversion is safe; clamp the rounded endpoint below one.
        u = self.probabilities.new_tensor(float(raw), dtype=torch.float64) / float(1 << 64)
        zero = u.new_zeros(())
        one = u.new_ones(())
        u = u.clamp(torch.nextafter(zero, one), torch.nextafter(one, zero))
        cdf = self.probabilities.to(torch.float64).cumsum(-1)
        cdf[-1] = 1.0  # The normalized distribution ends at exactly one.
        index = int(torch.searchsorted(cdf, u, right=False).item())
        if not 0 <= index < len(self.actions):
            raise ArithmeticError("Invalid categorical index; refusing to index outside action vocabulary")
        return self.actions[index]


class Engine:
    """Cache-free batched action scoring; no full [batch, sequence, vocabulary] logits."""

    def __init__(self, cfg: dict, budget=None, *, model=None, tokenizer=None, allow_cpu=False, pinned_revision=None):
        self.cfg, self.budget = cfg, budget
        self.device = torch.device(cfg["model"]["device"])
        if self.device.type != "cuda" and not allow_cpu:
            raise RuntimeError("Empirical runs require CUDA. CPU is only supported by the validation tests.")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Check the Lambda GPU image and PyTorch installation.")
            torch.cuda.set_device(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            reserve = int(cfg["compute"]["gpu_reserve_gib"] * (1 << 30))
            if reserve >= total:
                raise ValueError("Configured GPU reserve exceeds device memory")
            torch.cuda.set_per_process_memory_fraction((total - reserve) / total, self.device)
        self.batch_ceiling = cfg["compute"]["initial_batch_size"]
        self.oom_reductions: list[dict] = []
        self.forward_batches = 0
        self.forward_prompt_tokens = 0
        self.forward_seconds = 0.0
        self._capture, self._patches, self._saved = False, [], {}
        self._positions = None
        self.label_mass_reference = None
        self.provenance = {}
        if (model is None) != (tokenizer is None):
            raise ValueError("Inject both model and tokenizer for a validation test")
        if model is None:
            model, tokenizer = self._load(pinned_revision)
        self.model, self.tokenizer = model.eval(), tokenizer
        removed_vision_parameters = 0
        visual = getattr(self.model.model, "visual", None)
        if cfg["model"]["drop_unused_vision_tower"] and visual is not None:
            removed_vision_parameters = sum(p.numel() for p in visual.parameters())
            self.model.model.visual = None  # Text-only forward; config and LM head remain intact.
            del visual
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.text_config = getattr(model.config, "text_config", model.config)
        base = model.model
        self.body = getattr(base, "language_model", base)
        if not hasattr(self.body, "layers") or not hasattr(self.body, "embed_tokens"):
            raise TypeError("Model does not expose the supported Qwen text decoder layout")
        self.layers = self.body.layers
        if len(self.layers) != self.text_config.num_hidden_layers:
            raise ValueError("Layer count disagrees with the model configuration")
        types = getattr(self.text_config, "layer_types", None)
        self.layer_ids = [i for i in range(len(self.layers)) if types is None or types[i] == "full_attention"]
        if not self.layer_ids:
            raise ValueError("No full-attention layers available for the specified capture rule")
        self.lm_head = model.get_output_embeddings()
        self.embedding_rows = self.body.embed_tokens.weight.shape[0]
        self.output_rows = self.lm_head.weight.shape[0]
        self.hidden_size = self.text_config.hidden_size
        if self.lm_head.weight.shape[1] != self.hidden_size:
            raise ValueError("LM-head width disagrees with the text hidden size")
        if any(p.device != self.device for p in self.body.parameters()):
            raise RuntimeError("Text weights are offloaded or sharded. This runner requires one GPU with all text weights.")
        self.context_limit = min(cfg["compute"]["max_context_tokens"], self.text_config.max_position_embeddings)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define a padding or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not 0 <= self.tokenizer.pad_token_id < self.embedding_rows:
            raise ValueError("Padding token is outside the model's input embedding table")
        self.labels, self.label_ids = [], []
        for label in string.ascii_uppercase + string.ascii_lowercase + string.digits:
            ids = tokenizer.encode(label, add_special_tokens=False)
            if len(ids) == 1 and 0 <= ids[0] < min(self.embedding_rows, self.output_rows) and ids[0] not in self.label_ids:
                self.labels.append(label)
                self.label_ids.append(ids[0])
        if not self.labels:
            raise ValueError("Tokenizer has no validated single-token action labels")
        self.label_tensor = torch.tensor(self.label_ids, device=self.device, dtype=torch.long)
        self.head_rows = self.lm_head.weight.detach().index_select(0, self.label_tensor).contiguous()
        self.head_bias = None if self.lm_head.bias is None else self.lm_head.bias.detach().index_select(0, self.label_tensor)
        self.handles = [self.layers[layer].register_forward_hook(self._hook(layer)) for layer in self.layer_ids]
        self.provenance.update({"capture_layer_rule": "all_full_attention_layers",
                                "removed_unused_vision_parameters": removed_vision_parameters,
                                "capture_layer_indices": self.layer_ids, "embedding_rows": self.embedding_rows,
                                "lm_head_rows": self.output_rows, "context_limit": self.context_limit,
                                "action_labels": dict(zip(self.labels, self.label_ids)),
                                "cache_used": False, "action_distribution": "softmax conditional on available one-token action labels"})

    def _load(self, pinned_revision):
        from huggingface_hub import HfApi
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForConditionalGeneration
        mc = self.cfg["model"]
        token = os.environ.get("HF_TOKEN") or None
        revision = pinned_revision or HfApi(token=token).model_info(mc["id"], revision=mc["revision"]).sha
        config = AutoConfig.from_pretrained(mc["id"], revision=revision, token=token, trust_remote_code=False)
        if self.device.type == "cuda" and mc["require_fast_linear_kernels"] and config.model_type == "qwen3_5":
            for package in ("fla", "causal_conv1d"):
                if importlib.util.find_spec(package) is None:
                    raise RuntimeError(f"Missing {package} fast kernels. Run setup_lambda.sh; no silent slow-kernel fallback.")
            # Import now, so CUDA/ABI incompatibility is caught before loading 27B weights.
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
            from causal_conv1d import causal_conv1d_fn  # noqa: F401
        if config.model_type == "qwen3_5":
            cls = Qwen3_5ForConditionalGeneration
        elif config.model_type == "qwen3":
            cls = AutoModelForCausalLM
        else:
            raise ValueError(f"Unsupported architecture {config.model_type}; refusing a guessed layer layout")
        tokenizer = AutoTokenizer.from_pretrained(mc["id"], revision=revision, token=token, trust_remote_code=False)
        if self.budget:
            self.budget.check()
        model, loading = cls.from_pretrained(
            mc["id"], revision=revision, token=token, config=config,
            dtype=getattr(torch, mc["dtype"]), device_map={"": str(self.device)},
            attn_implementation=mc["attention_implementation"], trust_remote_code=False,
            output_loading_info=True,
        )
        text_missing = [x for x in loading.get("missing_keys", []) if "visual" not in x and "vision" not in x]
        if text_missing or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError(f"Incomplete checkpoint load: missing={text_missing}; mismatched={loading.get('mismatched_keys')}")
        versions = {}
        for package in ("torch", "transformers", "huggingface-hub", "safetensors", "accelerate",
                        "flash-linear-attention", "fla-core", "causal-conv1d", "triton"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.provenance = {"model_id": mc["id"], "resolved_revision": revision, "versions": versions,
                           "model_config": config.to_dict(), "unexpected_checkpoint_keys": sorted(loading.get("unexpected_keys") or []),
                           "device": str(self.device), "dtype": mc["dtype"],
                           "torch_cuda_version": torch.version.cuda,
                           "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU validation"}
        return model, tokenizer

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, layer):
        def hook(module, inputs, output):
            if self._positions is None:
                return output  # A reference forward outside Engine.score has no active edit.
            h = output if torch.is_tensor(output) else output[0]
            if h.ndim != 3 or h.shape[0] != len(self._patches):
                raise RuntimeError("Unexpected decoder output shape; aborting instead of indexing an incompatible tensor")
            rows = torch.arange(h.shape[0], device=h.device)
            if self._capture:
                self._saved[layer] = h[rows, self._positions].float().clone()
            selected = [i for i, patches in enumerate(self._patches) if layer in patches]
            if selected:
                row_ids = torch.tensor(selected, device=h.device, dtype=torch.long)
                delta = torch.stack([self._patches[i][layer] for i in selected]).to(device=h.device, dtype=h.dtype)
                if delta.shape != (len(selected), h.shape[-1]) or not bool(torch.isfinite(delta).all()):
                    raise ValueError("Invalid activation edit shape or nonfinite edit")
                # Fresh tensor prevents aliasing the captured baseline or another hypothesis branch.
                h = h.clone()
                h[row_ids, self._positions[row_ids]] += delta
                if torch.is_tensor(output):
                    return h
                return (h, *output[1:])
            return output
        return hook

    def prepare(self, job: Job):
        if not job.options or len(job.options) > len(self.labels):
            raise ValueError("Action menu is empty or larger than the validated label vocabulary")
        order = sorted(job.options, key=lambda action: stable_int(job.order_key, action))
        menu = "\n".join(f"{self.labels[i]}: {job.options[action]}" for i, action in enumerate(order))
        messages = [dict(x) for x in job.messages]
        messages.append({"role": "user", "content": "Choose your next action from the available tools.\n" + menu +
                         "\nReturn exactly its single action label. Do not add an explanation."})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False, preserve_thinking=False)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError("Empty model input")
        if len(ids) > self.context_limit:
            raise ContextOverflow(f"Context overflow for {job.key}: {len(ids)} > {self.context_limit}. No silent truncation.")
        if min(ids) < 0 or max(ids) >= self.embedding_rows:
            raise ValueError(f"Tokenizer/model mismatch for {job.key}: token outside embedding table")
        if any(layer not in self.layer_ids for layer in job.patches):
            raise ValueError("Requested edit at an uncaptured or nonexistent layer")
        return job, order, prompt, ids

    @torch.inference_mode()
    def _once(self, prepared):
        size = len(prepared)
        lengths = [len(x[3]) for x in prepared]
        width = max(lengths)
        ids = torch.full((size, width), self.tokenizer.pad_token_id, device=self.device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, item in enumerate(prepared):
            ids[i, :lengths[i]] = torch.tensor(item[3], device=self.device, dtype=torch.long)
            mask[i, :lengths[i]] = 1
        self._positions = torch.tensor(lengths, device=self.device, dtype=torch.long) - 1
        self._patches = [x[0].patches for x in prepared]
        self._capture = any(x[0].capture for x in prepared)
        self._saved = {}
        start = time.monotonic()
        try:
            output = self.body(input_ids=ids, attention_mask=mask, use_cache=False,
                               output_hidden_states=False, return_dict=True)
            last = output.last_hidden_state[torch.arange(size, device=self.device), self._positions]
            # Full vocabulary only at the LAST valid position: [B,V], never [B,T,V].
            # Its normalization exposes low label mass that a conditional softmax hides.
            full_logits = F.linear(last, self.lm_head.weight, self.lm_head.bias).float()
            if not bool(torch.isfinite(full_logits).all()):
                raise ArithmeticError("Nonfinite full-vocabulary action logits")
            logits = full_logits.index_select(1, self.label_tensor).to(torch.float64)
            counts = torch.tensor([len(x[1]) for x in prepared], device=self.device)
            valid = torch.arange(len(self.labels), device=self.device)[None, :] < counts[:, None]
            logits = logits.masked_fill(~valid, -torch.inf)
            probabilities = logits.softmax(-1)
            log_normalizer = torch.logsumexp(full_logits, dim=-1).double()
            masses = (torch.logsumexp(logits, dim=-1) - log_normalizer).exp().clamp(max=1)
            top_ids = full_logits.argmax(-1)
            top_is_available = ((top_ids[:, None] == self.label_tensor[None, :]) & valid).any(-1)
            if not bool(torch.isfinite(probabilities).all()) or bool((probabilities.sum(-1) <= 0).any()):
                raise ArithmeticError("Nonfinite or empty action distribution")
            features = torch.stack([self._saved[layer] for layer in self.layer_ids], dim=1) if self._capture else None
            results = []
            for i, (job, order, prompt, _) in enumerate(prepared):
                top_id = int(top_ids[i].item())
                mass = float(masses[i].item())
                audit = {"available_label_probability_mass": mass,
                         "available_action_count": len(order),
                         "unconstrained_argmax_token_id": top_id,
                         "unconstrained_argmax_token": self.tokenizer.decode([top_id]),
                         "unconstrained_argmax_is_available_label": bool(top_is_available[i].item()),
                         "control_label_mass_reference": self.label_mass_reference,
                         "below_control_label_mass_reference": mass < self.label_mass_reference if self.label_mass_reference is not None else None}
                results.append(Scored(order, probabilities[i, :len(order)].clone(), logits[i, :len(order)].clone(),
                                      features[i].clone() if job.capture else None, lengths[i], prompt, audit))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            seconds = time.monotonic() - start
            self.forward_batches += 1
            self.forward_prompt_tokens += sum(lengths)
            self.forward_seconds += seconds
            if self.budget:
                self.budget.last_batch_seconds = seconds
            return results
        finally:
            self._positions, self._patches, self._saved, self._capture = None, [], {}, False

    def score(self, jobs: list[Job]) -> list[Scored]:
        if not jobs:
            return []
        prepared = [(i, self.prepare(job)) for i, job in enumerate(jobs)]
        prepared.sort(key=lambda item: len(item[1][3]), reverse=True)
        results = [None] * len(jobs)
        cursor = 0
        while cursor < len(prepared):
            if self.budget:
                self.budget.check()
            count = min(self.batch_ceiling, len(prepared) - cursor)
            chunk = prepared[cursor:cursor + count]
            oom = False
            try:
                values = self._once([item[1] for item in chunk])
            except torch.cuda.OutOfMemoryError:
                oom = True
            # Leave the exception frame before retrying, releasing its intermediate tensors.
            if oom:
                if count <= 1:
                    raise RuntimeError("A single prompt does not fit GPU memory. Shorten the documented context budget or use more memory.")
                smaller = max(1, count // 2)
                self.oom_reductions.append({"attempted_batch": count, "next_batch": smaller,
                                             "longest_prompt": len(chunk[0][1][3])})
                self.batch_ceiling = smaller
                gc.collect()
                torch.cuda.empty_cache()
                print(f"GPU OOM: reducing action batch {count} -> {smaller}", flush=True)
                continue
            for (original, _), value in zip(chunk, values):
                results[original] = value
            cursor += count
        return results

    def telemetry(self):
        return {"successful_forward_batches": self.forward_batches, "scored_prompt_tokens": self.forward_prompt_tokens,
                "forward_seconds": self.forward_seconds, "initial_batch_size": self.cfg["compute"]["initial_batch_size"],
                "final_batch_ceiling": self.batch_ceiling, "oom_reductions": self.oom_reductions}
