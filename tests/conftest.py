import copy
from pathlib import Path

import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig

from bta.config import load_config, stable_int
from bta.runtime import Engine


class TestTokenizer:
    """Small deterministic tokenizer for unit tests, never used by the production loader."""
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"
    bad_token = False
    all_special_tokens = ['<pad>', '<eos>', '<|im_start|>', '<|im_end|>']

    def decode(self, ids):
        return ''.join(chr(i) if 32 <= i < 127 else f'<token_{i}>' for i in ids)

    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 and text.isascii() and text.isalnum():
            return [ord(text)]
        ids = [2 + stable_int(word) % 254 for word in text.split()]
        return ids + [256] if self.bad_token else ids

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


@pytest.fixture
def cfg(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/h100_qwen38.json")
    config["model"].update(device="cpu", dtype="float32", require_fast_linear_kernels=False)
    config["compute"].update(initial_batch_size=4, max_context_tokens=2048, gpu_reserve_gib=0)
    config["study"].update(fit_scenarios=2, calibration_scenarios=2, test_scenarios=2, max_rounds=2)
    config["output_root"] = str(tmp_path / "results")
    config["analysis"]["bootstrap_mc_standard_error"] = 0.1
    config['analysis']['permutation_mc_standard_error'] = 0.1
    return config


@pytest.fixture
def engine(cfg):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    text = Qwen3_5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=16, max_position_embeddings=2048, pad_token_id=0,
                            linear_num_key_heads=2, linear_num_value_heads=2,
                            linear_key_head_dim=8, linear_value_head_dim=8,
                            layer_types=["linear_attention", "full_attention"],
                            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                             "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3], "mrope_interleaved": True})
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config={"hidden_size": 32, "intermediate_size": 64,
                             "depth": 1, "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16,
                             "patch_size": 2, "temporal_patch_size": 1})
    config._attn_implementation = "sdpa"
    model = Qwen3_5ForConditionalGeneration(config)
    instance = Engine(cfg, model=model, tokenizer=TestTokenizer(), allow_cpu=True)
    yield instance
    instance.close()
