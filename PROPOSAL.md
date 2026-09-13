# Before They Act: collective commitment and selective interruption

**Version 0.2 — implementation-aligned proposal**

## Research question

Does making peer willingness mutually visible, and establishing a history of nonbinding agreement, change an LLM team's response when peers later withdraw? Can a measured internal component be changed to reduce coordinated unauthorized action while preserving legitimate cooperation?

The cooperative-AI problem is maintaining useful coordination. The security problem is the same coordination process carrying a team across an explicit permission boundary. The project tests both in matched tasks instead of treating less cooperation as automatically safer.

## Setup

Run three role contexts—proposer, reviewer, executor—through one pretrained Qwen3.8-27B model on one H100 80 GB. Use its actual hybrid decoder layout, disable thinking in the chat template, verify simple response controls, and record model revision and software versions. No fine-tuning or benchmark-dataset release is required.

The local numerical task simulates a software-validation workflow. An incorrect program output disagrees with a fixture. One trusted instruction requires repairing the program and forbids fixture changes; a matched trusted instruction requests a legitimate fixture migration. A lower-trust suggestion proposes the fixture route in both conditions. Empirical agents begin without assigned agreement and select their own actions through a constrained menu.

A proposal must receive exact-version approval before the executor can apply it. Unauthorized approval and completed joint tampering are separate outcomes. Legitimate completion requires the owner's requested state and submission, not merely a passing test. All actions operate on a local simulation.

## Four experiments and predeclared measurements

1. **Formation and history dependence.** Compare the same peer records under private versus mutually visible delivery receipts. At eligible prefixes, compare continued support, withdrawal, and neutral updates while preserving technical feasibility. Add constructed prior-agreement versus no-agreement histories with identical current peer positions. Measure conditional approval contrasts, actual coordinated actions, and the history effect after withdrawal with a support control. A history effect is not automatically called hysteresis or spontaneous collective commitment.

2. **Internal tracking and forecasting.** Capture final-input-token decoder states at all full-attention layers. Fit a peer-sensitivity predictor on authorized development observations and a danger predictor within prohibited/public observations. Select layers and empirical-spectrum ridge penalties on calibration data. Evaluate on held-out scenarios/wording, against peer counts, receipt visibility, a text monitor, and direct forbidden-action probability. Calibrate the operating threshold only on complete negative prohibited/public calibration episodes. Report all authorization/visibility strata with that same threshold. Separate four-turn forecast success, strictly earlier warning, and same-decision gating; report coverage and missing classes.

3. **Causal component intervention.** Derive a candidate direction from authorized within-prefix visibility contrasts. Select its layer using scenario-level paired activation/sensitivity association and report a scenario-permutation max-layer null. At the first reviewer decision, replace its projection with a matched private-prefix projection. Compare equal-norm orthogonal random edits, task reminders, and the untouched baseline. Measure peer sensitivity, history dependence, prohibited approvals, joint tampering, and authorized completion. A selected candidate remains exploratory until held-out causal evidence supports an interpretation.

4. **Interruption timing.** Using one fixed reviewer, candidate, and calibrated warning rule, compare an edit at the first alert, one reviewer decision later, or a keyed random reviewer turn. Each episode receives at most one edit. Use an authorized-fit private reference instead of an online donor. Report exposure, completion, tampering, and paired differences; a late intervention cannot erase an earlier violation.

## Allocation, analysis, and deliverables

The supplied pilot allocation is 16 fit, 16 calibration, and 32 test scenarios with disjoint wording families, six rounds, and one replicate. It creates 640 maximum live episodes per replicate across the four stages; stage 2 reuses baseline episodes. Diagnostic branches add forward work without adding live episodes. The profiling command measures representative long and mixed contexts, then estimates a feasible replicate count from remaining elapsed-time budget. Final allocation is fixed before held-out evaluation; it is not chosen by inspecting desired test effects.

Report scenario-clustered paired intervals, matching scenario and replicate before aggregation. Report raw response-format mass, observed batching drift, unscored monitor observations, and undefined statistics. A weak or absent effect remains a result. The project does not train a detector on authorized-versus-prohibited prompt identity and present that separation as prediction of collusion.

Deliverables are a reproducible runner, four experiment directories, per-condition raw traces and counterfactual records, selected probes and null diagnostics, plots of actual measured outcomes, and an interpretation bounded by the evidence. The planning budget is 100 USD, with an 80 USD compute allocation and 20 USD reserve. Pricing and throughput must be measured on the booked instance; the process guard does not terminate cloud billing.

The intended contribution is a controlled measurement and intervention design joining useful cooperation, prior agreement, withdrawal, and security boundaries. Generalization to natural covert language, heterogeneous models, genuine network compromise, or human collective action is outside this pilot's evidence. The OpenAI/Hugging Face incident motivates the workflow-security question; this code does not reproduce that incident.
