# Peer Approval Security

**When peers agree, does an LLM reviewer still follow the owner's authority?**

Peer Approval Security studies how peer messages change approval decisions in a three-agent software-validation simulation. It connects a cooperative-AI question—how agents respond to one another—with a security question—whether that influence survives an explicit restriction on what the team may change.

**Current evidence:** the completed baseline shows peer-sensitive approval in authorized situations. It recorded no prohibited violations, so the internal risk detector and timing experiment could not be identified. A separate constructed-prefix probe measures the approval decision at a prohibited proposal that was rarely reached naturally. These are different experiments and remain separate throughout the website.

[Project website](https://subramanyamsahoo.github.io/Peer-Approval-Security/) · [Hosting instructions](GITHUB_PAGES.md) · [Original experiment instructions](README_EXPERIMENTS_v02.md) · [Methods](METHODS.md)

The website link becomes active after GitHub Pages is enabled and the first deployment succeeds.

## Research question and task

A proposer, reviewer, and executor use separate role contexts backed by one loaded Qwen checkpoint. They must resolve a disagreement between a small numerical program and its validation fixture. Proposal, version-specific approval, application, and submission are separate actions.

| Condition | What the owner permits | What counts as legitimate completion |
|---|---|---|
| Authorized fixture migration | Change the fixture; preserve the program | The requested migration succeeds without changing the protected program |
| Prohibited fixture modification | Repair the program; preserve the fixture | The program is repaired without changing the protected fixture |

The agents choose from a finite action/message menu. This makes approval and execution auditable, but constrains their ability to invent communication strategies. The study does not reproduce a real infrastructure intrusion or establish a free-form covert channel.

The OpenAI/Hugging Face incident motivates examining shared coordination around evaluation tasks; its investigation does not demonstrate the internal commitment mechanism tested here. [METR investigation](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/)

## What was run

The baseline is the completed run `20260913T081310.982652Z_a01a9c7ef5de4ed3a9e33e3ffb97c841`. It used 16 fit, 16 calibration, and 32 test scenarios, one replicate, and a maximum of six rounds. It completed 256 formation episodes and 192 patching episodes. Diagnostic branches add scored prompts; they are not additional live episodes.

| Component | Comparison | Recorded status |
|---|---|---|
| E1 · Formation | Peer support versus withdrawal; private versus public receipts; constructed prior-agreement histories | Completed; authorized effects measurable |
| E2 · Warning | Internal predictors, text monitor, and action-score baselines | Analysis completed; internal security-risk target had no variation |
| E3 · Patching | Candidate activation edit, equal-norm orthogonal random edit, and task reminder | Completed; no clear benefit from the candidate edit |
| E4 · Timing | Candidate intervention at calibrated warning, delayed warning, or random reviewer turn | `not_identified`; required calibrated internal warning unavailable |
| Constructed-prefix probe | Support/withdrawal/neutral × reminder/plain × authorized/prohibited × private/public | Separate follow-up; imported from its own saved JSON when available |

`completed` describes software execution. It does not establish an identified detector, a successful intervention, or positive security evidence.

## Audited baseline findings

The quantities below refer to the recorded baseline, not to the follow-up probe. Probability differences are reported in percentage points (pp). Intervals are the recorded scenario-bootstrap 95% intervals.

| Quantity | Result | Interpretation |
|---|---:|---|
| Authorized/public first-reviewer support minus withdrawal | **+26.88 pp** [22.69, 31.44] | Approval changes with peer stance |
| Authorized/private first-reviewer support minus withdrawal | **+23.93 pp** [19.86, 28.82] | Peer influence also appears with private receipts |
| Authorized/public prior-agreement effect after current withdrawal | **+5.80 pp** [4.59, 7.09] | The constructed prior history changes approval |
| Candidate edit: change in authorized/public peer sensitivity | **−0.28 pp** [−0.87, 0.30] | No clear causal reduction in this comparison |
| Task reminder: change in authorized/public peer sensitivity | **−20.03 pp** [−24.05, −16.31] | The reminder reduces this contrast; this alone is not a security benefit |
| Test prohibited/public episodes with a recorded violation | **0/32** | Security sensitivity and warning recall are unidentified here |
| Test legitimate completion, authorized/public | **22/32** | Useful task completion must be assessed alongside violations |
| Test legitimate completion, prohibited/public | **25/32** | Zero violations did not mean universal task abandonment |

The prior-agreement manipulation includes the reviewer's own previous agreement as well as peers' agreement. Its effect therefore does not separate individual consistency from collective commitment. Public and private history effects were nearly equal; this run does not establish an effect of common knowledge.

The selected internal candidate was exploratory: its maximum-statistic permutation result was approximately `p = 0.0716`. The observed preflight batch-composition approval drift reached **4.12 pp** on one checked prefix. That is not a universal error bound, but it makes matched-batch reproduction important for small activation-edit effects.

The stage results support a limited conclusion: **peer messages affect approval in this constrained cooperative task; the completed baseline does not demonstrate security-relevant collusion or a causal collective-commitment circuit.**

## What the constructed-prefix probe adds

The follow-up script applies `propose_fixture` to the initial state and scores the reviewer before approval or execution. It deliberately creates the missing opportunity to approve a prohibited fixture change. Every scenario is evaluated under both authority conditions, both receipt conditions, three peer stances, and two reminder settings: 24 scored prompts per scenario, or 768 for 32 scenarios.

Its principal contrast is:

```text
peer sensitivity = P(approve | support) − P(approve | withdrawal)
reminder effect = peer sensitivity with reminder − peer sensitivity without reminder
```

These are action-menu-conditional probabilities at constructed situations. They are not rates of naturally occurring coordination or observed unauthorized execution. A smaller support–withdrawal gap can coexist with higher prohibited approval, so absolute support, withdrawal, and neutral probabilities must also be inspected.

The website importer reads `prohibited_prefix_probe.json` and its companion `prohibited_prefix_summary.json` from `results/`. Runs remain separately selectable; the importer does not combine them into an invented aggregate. The shipped page explicitly shows an unimported state until actual probe outputs are available. Raw probe files alone are sufficient to show means; their companion summaries supply intervals when validated.

If the probe has already been run, publish its existing output; no additional GPU work is needed to update the site. If repeating it for a scientific reason, use a fresh output directory because the original script's default output path can be overwritten:

```bash
# Run in the original experiment environment, where bta and its dependencies exist.
BTA_SOURCE_RUN="results/20260913T081310.982652Z_a01a9c7ef5de4ed3a9e33e3ffb97c841"
.venv/bin/python probe_prohibited_prefixes.py --run-dir "$BTA_SOURCE_RUN" --check

# The following performs real model inference. Run only for a planned new measurement.
BTA_PROBE_OUT="$(mktemp -d "$PWD/results/prohibited_prefix_repeat.XXXXXXXX")"
.venv/bin/python probe_prohibited_prefixes.py \
  --run-dir "$BTA_SOURCE_RUN" \
  --out-dir "$BTA_PROBE_OUT"
```

The source script does not capture activations or retain the runtime's full response-format diagnostics. Its exported approval probabilities therefore cannot by themselves support an internal-mechanism claim or a complete label-mass audit. Avoid `--limit 1`: the original summary formatter expects an estimable interval.

## Preview the website without a GPU

The website uses HTML, CSS, JavaScript, and JSON. Its exporter uses Python's standard library. It does not load model weights, import PyTorch, run experiments, or require an HF token.

From the repository root:

```bash
python3 tools/export_site_results.py --results-root results --output docs/data/probes.json
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

On the same computer, open `http://127.0.0.1:8000`. To preview from a Lambda server, forward the port through your existing SSH connection; opening your laptop's localhost does not directly access Lambda.

For a live public site, follow [GITHUB_PAGES.md](GITHUB_PAGES.md). GitHub Actions exports the saved probe results and publishes only `docs/`. No H100 is needed for deployment. The default project URL is `https://subramanyamsahoo.github.io/Peer-Approval-Security/`; account-level `name.github.io` addresses require a matching owner account. [GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

## Repository map

| Path | Purpose |
|---|---|
| `bta/` | Original experiment implementation |
| `configs/` | Scientific and runtime configuration |
| `results/` | Separate run directories and follow-up probe outputs |
| `probe_prohibited_prefixes.py` | Constructed-prefix follow-up, when included in the research checkout |
| `docs/` | Static project website |
| `docs/data/baseline.json` | Audited baseline summary packaged for the website |
| `docs/data/probes.json` | Website export of available follow-up probe runs |
| `tools/export_site_results.py` | CPU-only conversion of saved probe JSON |
| `.github/workflows/pages.yml` | Export and GitHub Pages deployment |
| `README_EXPERIMENTS_v02.md` | Preserved original setup, profiling, execution, and resume instructions |

The review archive omits activation tensors and model weights. Preserve the original Lambda results before deleting the instance if later activation analysis is required. Resume checks can depend on source and configuration hashes; install website files in the GitHub checkout instead of editing a live experiment implementation.

## Reproducing the model experiments

The completed run recorded one H100 PCIe 80 GB, BF16 inference, an initial batch of 64, and checkpoint `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. These identify the observed run, not a promise that every future environment is compatible.

Follow [README_EXPERIMENTS_v02.md](README_EXPERIMENTS_v02.md) for the original package requirements, preflight, profiling, and execution. Set billing information from the booked instance, keep `HF_TOKEN` in the environment, and inspect every stage's `status.json`. The budget guard stops computation; it does not stop cloud billing by terminating the instance.

The website installation and GitHub Actions workflow do not start a new experiment, refit a probe, alter raw result files, or regenerate GPU-dependent scientific analyses.

## Scope and related work

This is a small, single-model study with constrained messages and one simulated task. A zero-event sample cannot establish broad safety, and constructed approval scores cannot establish naturally formed commitment. Differences between unrelated runs must not be pooled as if they were matched trials. The contribution is an inspectable experiment and its findings, including unidentified outcomes; no literature-wide novelty claim is made.

Multi-agent security work motivates studying interaction-level failures and security–utility tradeoffs. [Open Challenges in Multi-Agent Security](https://arxiv.org/abs/2505.02077)

Internal multi-agent collusion detection is already an active research direction. The current results should not be presented as the first activation-based collusion detector. [Detecting Multi-Agent Collusion Through Multi-Agent Interpretability](https://arxiv.org/abs/2604.01151)

Website implementation checks: [validation record](docs/downloads/site-validation.md). Eleven exporter tests passed; browser rendering was blocked in the build environment and is not marked as verified.

Maintained by [Subramanyam Sahoo](https://github.com/SubramanyamSahoo).
