"""Opt-in small-file integration check: no model weights are downloaded."""
import os
import string

import pytest


@pytest.mark.skipif(os.environ.get('BTA_TEST_HF_TOKENIZER') != '1', reason='Set BTA_TEST_HF_TOKENIZER=1 to check the real public HF tokenizer')
def test_actual_qwen38_nonthinking_template_and_label_bounds():
    from huggingface_hub import HfApi
    from transformers import AutoConfig, AutoTokenizer
    model = 'Qwen/Qwen3.8-27B'
    token = os.environ.get('HF_TOKEN') or None
    revision = HfApi(token=token).model_info(model, timeout=30).sha
    config = AutoConfig.from_pretrained(model, revision=revision, token=token, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, token=token, trust_remote_code=False)
    assert config.model_type == 'qwen3_5'
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Which is 2 + 2? A: 4. B: 5. Return one label.'}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
    assert '<think>' not in prompt or prompt.rfind('</think>') > prompt.rfind('<think>')
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert 0 <= min(ids) <= max(ids) < config.text_config.vocab_size
    labels = [tokenizer.encode(s, add_special_tokens=False) for s in string.ascii_uppercase]
    assert all(len(ids) == 1 and 0 <= ids[0] < config.text_config.vocab_size for ids in labels)
