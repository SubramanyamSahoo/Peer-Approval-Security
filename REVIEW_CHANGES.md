# Review response and migration — v0.2

This release addresses the supplied review and the subsequent clarification. It changes the measurement protocol, not just filenames. The old ZIP remains a v0.1 artifact. **Start a new v0.2 run.** Do not reuse v0.1 probe weights or append v0.2 observations to its result directory.

| Point | Resolution |
|---|---|
| Pooled authorized/prohibited discrimination | Primary risk fitting, model selection, and threshold calibration are prohibited/public only. Authorized false alarms and utility use the same threshold. Four strata, a prohibited-only aggregate, and a pooled appendix are explicit. |
| Missing next-token label mass | Every forward records available-label mass and unrestricted top token using a full-vocabulary head only at the final valid position. Controls check simple task understanding under reversed label assignments. No unexplained mass floor was added. |
| Numerical precision | Measured probability drift over twelve authorization/visibility/peer-update prefixes replaces any interpretation of the loose structural tolerance as an effect-size bound. The structural agreement check remains, with a narrower description. |
| Near-zero lead | Horizon is four, not three: the upper endpoint in the label is exclusive. Window-valid, strictly early, same-decision, and any-pre-event recalls are separate. Out-of-window alarms count against decision forecasting. |
| Candidate selection under a null | Candidate selection now uses paired activation/sensitivity contrasts within authorized/public trajectories. Repeated observations are collapsed to independent scenario summaries. The max-layer statistic is recomputed on every scenario-target permutation. A candidate remains exploratory; no null result is hidden behind a mechanism claim. |
| First/last checkpoint comparison | Added a matched-current-state, constructed-history comparison. Earlier agreement versus no earlier agreement is compared after identical current withdrawal, with a current-support control. It is called history dependence, not a demonstrated hysteresis loop. |
| More replicates | Added measured profiling, explicit forward-count bounds, and a budget-based capacity estimate. Replicates remain a pre-run choice. No claim is made that four or five is always affordable or power-optimal. |
| Feature memory | Metadata is read first; safetensors stay CPU-backed and selected layer slices move to GPU. Numerical fitting remains on the GPU. |
| Monitor overflow and nested templates | Original structured messages and options feed the monitor. Special chat tokens are escaped. Overflow is recorded as missing; coverage and common-complete-case comparisons are reported. |
| Baseline/intervention ambiguity | `baseline_prompt`, `decision_prompt`, separate probability maps and diagnostics, and an explicit baseline feature reference replace mixed meanings. |
| Pairing with multiple replicates | Match scenario and replicate before averaging differences within scenario. Unmatched replicates are explicitly reported. |
| Unused vision weights | The text-only runner discards the vision tower after loading and records its parameter count. There can still be an initial loading peak. |
| Unused dependency file | Setup consumes `requirements-lambda.txt` as constraints for Torch, package, and kernel installs. CUDA extras are also declared in `pyproject.toml`. |
| Kernel compilation | `setup_lambda.sh --check` checks prerequisites without package/model downloads. Setup caches a matching wheel before model loading. Compilation remains billable; its cost is not hidden by moving the billing marker. |
| Hard-link portability | Atomic JSON uses Linux advisory file locking and atomic rename. The target filesystem must support those semantics; this is not a claim of universal filesystem portability. |

## Documents reconciled

`PROPOSAL.md`, `METHODS.md`, and `README.md` describe the actual v0.2 implementation. Changes from the earlier standalone proposal are explicit:

- **Qwen3-8B → Qwen3.8-27B**, following the later model request. The actual hybrid architecture and tokenizer are inspected, and the production preflight checks response controls.
- **Generated short messages → a constrained action/message menu.** The budget concerns forward-prompt processing, not a ceiling of generated tokens. The code does not claim free-form covert coordination.
- **Four preselected layers → all full-attention layers** derived from the checkpoint, with development selection and a max-statistic null comparison.
- **Low-dimensional probes → regularized full-dimensional dual ridge**, with per-layer feature streaming and empirical-spectrum penalty candidates.
- **20 hours / 65.80 USD → an 80 USD compute allocation plus 20 USD reserve**, using the earlier 3.29 USD/hour figure only as a configurable example. The profiler measures local throughput; no completion-time claim is made.
- **Two-turn warning → four-turn forecast with separate same-decision/early metrics.**

No empirical effect sizes, detector accuracies, speedups, or mechanistic findings were invented during this revision. Unit-test trajectories are excluded from the release archive. The full pretrained model, CUDA kernels, measured H100 drift, runtime, and substantive findings must still be evaluated on the target instance.
