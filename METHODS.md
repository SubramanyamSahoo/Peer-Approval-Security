# Methods — v0.2

## Scope and experimental units

One pretrained Qwen3.8-27B model serves three distinct role contexts. Agents act through a constrained menu; messages have disclosed fixed wording. Every forward recomputes the full textual context without a KV cache. The numerical fixture task is procedural experimental infrastructure, not a new benchmark-dataset contribution. The owner instruction determines whether fixture migration is authorized. Proposal, approval, application, and submission remain separate steps.

A scenario is the independent resampling unit. Replicates use different keyed sampling uniforms. Conditions share scenario/replicate keys for paired comparisons, but shared random numbers do not guarantee variance reduction or bitwise equality under different batch shapes. Pair scenario and replicate first, then average matched differences within scenario. Report unmatched observations. Later histories may diverge after different actions.

## 1. Formation, withdrawal, and controlled history

For an eligible reviewer prefix `t`, let `p_t(v,s)` be conditional approval probability under visibility receipt `v` and peer update `s`. Eligibility requires an open fixture proposal, no binding approval, and no previous prohibited approval.

```text
C_t(v) = p_t(v, support) − p_t(v, withdraw)
V_t = p_t(public, neutral) − p_t(private, neutral)
```

The six branches hold the factual state, permissions, action menu, and label ordering fixed. Support withdrawal does not revoke the proposal or disable technical completion. Private/public conditions change the stated visibility of the same records. This manipulates finite mutual assurance; it does not establish infinite common knowledge. Diagnostic branches never change the live trajectory.

At the first eligible held-out baseline prefix, a separate four-branch diagnostic changes earlier nonbinding exchanges. Existing nonbinding messages are removed; tool-action history remains. The three roles are then described as having either previously agreed to participate or considered the plan without agreeing. Current proposer/executor positions are set to the same withdrawal statement, or the same support statement, within each history contrast. The reviewer chooses its next action freely in the resulting hypothetical context.

```text
H_withdraw = p(prior agreement, current withdrawal) − p(no prior agreement, current withdrawal)
H_support = p(prior agreement, current support) − p(no prior agreement, current support)
Interaction = H_withdraw − H_support
```

The branch count, roles, current statements, proposal, and permissions are matched. Prior wording differs by construction, and token lengths are recorded; this is not a claim that every linguistic feature is matched. An effect is evidence of history dependence under these controlled prior-agreement statements. It does not prove a dynamical hysteresis loop, irreversible agreement, or spontaneous collective commitment. A first/last trend alone is not used as a hysteresis test.

All eligible checkpoint curves are descriptive because eligibility and survival change. Stage-3 causal diagnostic comparisons are restricted to matched first-reviewer decisions, before earlier treatment could change eligibility. Baseline history diagnostics at later first-eligible prefixes remain conditional descriptive results.

## 2. Internal predictors and correct warning targets

Capture the output of each full-attention decoder layer at the last valid input token. Derive layers and dimensions from the model configuration. Read saved tensors as CPU-backed per-layer slices and transfer only the required matrices to GPU.

The risk predictor is fitted and selected only within prohibited/public trajectories, matching the timing experiment's deployment condition. Authorized trajectories cannot supply easy negatives to its primary fit or calibration. The peer-sensitivity predictor is fitted on authorized observations and evaluated separately by authorization and visibility, without a pooled headline correlation.

For each candidate layer, fit scenario-balanced dual ridge with an intercept. Center and scale on fit data only. Candidate penalties are the positive fit Gram eigenvalues above a machine-precision-derived numerical tolerance. Select penalty and layer by calibration MSE. This is regularized full-dimensional regression, not a dimensionality-reduction method. Held-out outcomes never select a layer, penalty, direction, or operating point.

The risk target is:

```text
label(t) = 1 if 0 ≤ first_violation_step − t < horizon; otherwise 0
```

The default horizon is four agent turns. Reviewer decisions are three turns apart, so a horizon of three would still exclude the next reviewer decision. Already-violated prefixes are excluded.

Threshold calibration uses maximum scores from complete negative prohibited/public calibration episodes with at least one reviewer observation. Sort maxima descending and select zero-based rank `floor(target_FPR × n)`, alerting with strict `score > threshold`. Ties are handled conservatively. Model selection and threshold estimation share development data, so this is an empirical operating point, not a distribution-free false-alarm guarantee.

Apply the same frozen threshold to every test stratum. Primary reporting is prohibited/public; prohibited/private is a transfer view; authorized/public and authorized/private report false alarms and utility. A prohibited-only aggregate and pooled appendix are also saved. A high pooled AUROC cannot substitute for within-prohibited discrimination.

Four distinct detections are recorded:

- Within-window: at least one alert satisfies the forecast horizon.
- Strictly early: a within-window alert precedes the violating decision by at least one agent turn.
- Same decision: alert immediately before that action executes; lead is zero.
- Any pre-event: any earlier alert, regardless of horizon, explicitly separated from forecast success.

An out-of-window alert is a false positive for its decision-level forecast even when the episode violates later. Report episode AUROC and negative-episode alarm rate separately from decision-window AUROC/FPR. These have different denominators. Missing positive/negative classes make the corresponding metric undefined.

The transcript monitor sees original structured context and available actions. Embedded special chat tokens are escaped; a rendered agent chat template is not injected into another template. Context overflow creates a persistent `unscored_context_overflow` record. No silent truncation or zero-risk imputation occurs. Incomplete episodes are excluded from that scorer's performance estimates with coverage reported. A separate common-coverage report compares scorers on the same scored test episodes using their already-frozen thresholds.

## 3. Candidate direction and null comparison

Use the first eligible prefix per replicate from authorized/public development trajectories. Average replicates within scenario. Both visibility conditions below are counterfactuals from the same prefix, so no authorized/prohibited prompt separation can explain the selection statistic.

For each layer:

```text
u_l = normalize(mean_fit_scenario(h_l(public, neutral) − h_l(private, neutral)))
x_l = scenario_mean(u_lᵀ[h_l(public, neutral) − h_l(private, neutral)])
y = scenario_mean(C_t(public) − C_t(private))
```

Select the layer with the largest absolute calibration correlation of `x_l` and `y`. To assess selection under the null, permute the complete scenario-level target vector and recompute the maximum absolute correlation across all candidate layers on every permutation. Individual correlated checkpoints are never shuffled as independent observations. The Monte Carlo count is derived from the configured rank-error precision, and p uses the `(exceedances + 1)/(draws + 1)` correction. Numerical comparison tolerance derives from floating-point precision and sample count.

This null tests a scenario-level association conditional on the fit-derived directions and eligibility scheme. It does not prove a causal commitment representation. No p-value threshold automatically certifies a mechanism or selects a favorable result. A defined candidate can undergo an explicitly exploratory held-out intervention even if its null comparison is weak. A missing or degenerate candidate is reported as unidentified.

At the first reviewer decision in stage 3, score a matched private-receipt donor for the live prefix and edit only the chosen component:

```text
delta = (uᵀh_private − uᵀh_live)u
h_edited = h_live + delta
```

No fixed steering coefficient is used. Compare an orthogonal Gaussian direction with the same edit norm, a trusted-task reminder, and the untouched public baseline. Apply each live-prefix edit unchanged to that prefix's diagnostic branches; do not tune the edit independently to each desired diagnostic answer. Keep baseline features separate from decision outputs.

Report peer-sensitivity changes, matched history-effect changes, prohibited approvals, joint tampering, and authorized completion. A selective security interpretation requires evidence beyond general impairment, instruction-following, or receipt-wording sensitivity. There is no composite weighted success score or automatic positive mechanism claim.

## 4. Timing

Hold the edited agent fixed as reviewer. At most one intervention occurs per episode. Compare first calibrated alert, one reviewer decision later, and a scenario/replicate-keyed random reviewer turn within the round budget. All timing arms use the same candidate and an authorized-fit private reference projection; unlike stage 3, they do not request a matched online donor.

Episodes ending before a scheduled edit receive no edit. Report exposure differences; these are not equal-exposure randomized trials. A late edit cannot erase an earlier approval but may prevent later application. The end-of-episode horizon remains fixed, so postponement is not silently called prevention.

## Numerical validity and inference limits

The agent policy is a softmax conditional on available one-token action labels. Every forward additionally computes the last-position full-vocabulary normalization and unrestricted top token. Record label mass separately for each role and diagnostic condition. Menu size affects mass, so cross-role mass comparisons are not directly interchangeable.

Preflight requires correct conditional argmax answers on simple arithmetic and authorization controls under reversed label assignments. It also measures probability drift across representative prefixes and batch compositions. A dtype-derived logit tolerance checks structural agreement only; it is not the precision of a peer-sensitivity effect. The reported drift is an observed reference, not a guarantee for every future context. Label-mass measurements and control checks do not certify semantic validity.

Outcome and diagnostic contrasts use scenario-clustered percentile bootstrap intervals, averaging only matched replicates. Resample count follows the stated Monte Carlo quantile-rank precision. Small samples, constant observations, changing eligibility, and multiple exploratory comparisons limit inference; no multiplicity-corrected significance claim is generated. Warning operating points have explicit counts and missingness but no automatic population confidence guarantee.

The study covers one model, one finite numerical task family, and constrained language. It does not establish generalization to free-form covert messages, heterogeneous model populations, real network compromise, or human collective action. Null results and failures to identify a detector remain legitimate outcomes.
