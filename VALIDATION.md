# Delivery validation — v0.2, 2026-09-13

**No H100 was available in the authoring environment. The full pretrained Qwen3.8-27B has not been run here. The ZIP contains no empirical findings or test-generated research figures.**

The offline suite passes **43 tests**. A separate opt-in integration test against the actual public Hugging Face Qwen3.8-27B configuration and tokenizer also passes: **44 passing tests in total**, with the network test skipped in the default offline invocation.

The validation environment used Python 3.12.14, Torch 2.10.0+cpu, Transformers 5.17.0, huggingface-hub 1.31.0, safetensors 0.8.0, Matplotlib 3.11.2, Accelerate 1.15.0, and pytest 9.1.1. The offline model tests use the real Transformers hybrid Qwen decoder at a tiny, randomly initialized size. They use a deterministic test tokenizer; production never loads that tokenizer.

## What was exercised

- Last-position full-vocabulary scoring agrees with a full conditional-model forward pass. Unequal-length padding, batch reordering, zero edits, and edits that change action probabilities are covered.
- Invalid token IDs, excessive context, invalid layer indices, categorical sampling bounds, and simulated GPU OOM are checked. OOM backoff preserves job order and keyed random draws.
- Authorized maintenance, prohibited tampering, legitimate repair, and exact-version approvals follow the environment rules. Withdrawal preserves the proposal and available tools.
- The warning regression fixture deliberately produces an authorization confound: prohibited-task AUROC remains 0.5 while the pooled statistic is higher. Calibration excludes authorized and private episodes from the public prohibited-task deployment scope.
- Same-decision, strictly earlier, and horizon-valid alerts are separated. The four-decision horizon and out-of-window false alerts are tested directly.
- Low native label probability is exposed even when conditional label probabilities appear confident. Reversed-label semantic controls and a failing semantic preflight are exercised.
- Text-monitor overflow produces an explicitly missing observation; incomplete coverage is reported. Special chat-control tokens in monitored material are escaped.
- Activation files are sliced by layer before transfer to the analysis device. Scenario-level permutation inference recomputes the maximum over candidate layers and handles numerical ties.
- Prior agreement and never-agreed history branches share the same present peer positions and tool state. These constructed histories test history dependence; they do not establish spontaneous commitment or a hysteresis loop.
- Paired intervention comparisons match both scenario and replicate before averaging within scenario.
- Atomic writes, immutable committed steps, interrupted-run recovery, unique runs, and unchanged resume configuration are covered. JSON lock files have a separate extension so report readers do not parse them as results.
- Runtime profiling has its own result kind and cannot be resumed into an empirical run. Profiling does not silently change the replicate count.
- An end-to-end formation → learning → patching → timing → reporting pipeline runs, including explicit handling of unidentified mechanisms and a resume check.

The end-to-end test **forces selected actions solely to exercise positive and negative paths**. Its model forward passes, diagnostic probabilities, fitted tensors, patches, and file operations are real software operations. Its outcomes are test fixtures, not behavior measured from the pretrained checkpoint.

## Real Hugging Face tokenizer integration

The opt-in test resolves an immutable revision of `Qwen/Qwen3.8-27B`, downloads only its configuration and tokenizer, and checks the `qwen3_5` model type, non-thinking chat template, prompt token bounds, and single-token A–Z action labels. IDs are derived at runtime; none are hardcoded into the runner. The empirical loader separately records the immutable revision that it actually loads.

```bash
python -m pytest -q
BTA_TEST_HF_TOKENIZER=1 python -m pytest -q tests/test_hf_tokenizer.py
```

Editable installation, dependency consistency, shell syntax, and command-line workload planning were also checked. At the supplied one-replicate configuration, planning derives 256 baseline episodes, at most 192 patching episodes, and at most 192 timing episodes. The forward-job planning bound is 30,976, including diagnostic branches and monitor calls; this is a workload count, not a measured cost or completion-time guarantee.

## What still needs the Lambda H100

Full checkpoint loading, initial memory peak before the unused vision tower is dropped, BF16 behavior at 27B scale, CUDA kernel compatibility, actual mixed-length GPU execution, effective batch size, throughput, total cost, and all scientific findings remain unverified here.

The production preflight runs on your instance before collecting episodes. It checks actual-model structural consistency and semantic controls, records native label mass and observed probability drift, and stops on failure. The CPU test-only structural bypass is not exposed by the production CLI. The profiler then measures representative workloads on that GPU; its cost estimate remains a planning estimate. Neither a budget stop nor script completion terminates the rented instance.
