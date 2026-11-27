# Local Potential Field Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add local exponential scan fields, physically feasible experts and separate recurrent policies with explicit radar commands.

**Architecture:** Opt-in field-v2 observation/action protocol alongside the preserved legacy baseline. Environment owns timestamp-derived local fields and public goals; both experts and networks consume the same arrays.

**Tech Stack:** Python, Gymnasium, NumPy, PyTorch, TensorBoard, native JavaScript Canvas, Matplotlib.

**Spec:** docs/superpowers/specs/2026-11-27-local-potential-field-design.md

## Global Constraints

- Source stays in src; all output/cache in outputs; TF events in tf-logs; all tunable values exposed through run.sh.
- Preserve two independent networks and datasets, original finite-response kinematics and SI units.
- obs-field-v2 has vector[111/134], potential[4,33,33], field_context[16]; action[3] = unit world XY + relative normalized radar pan.
- Default Config legacy compatibility; run.sh explicit field switch selects the new protocol. Never load v1 model into v2 env silently.

### Task 1: Timestamp field and physical environment protocol

Files: src/potential_field.py (new), src/config.py, src/observations.py, src/env_base.py, src/env_coverage.py; tests/test_potential_field.py (new).

Produces opt-in `Config.potential_enabled`, `potential_radius`, `potential_period`, `potential_decay`, `coverage_pan_limit`, `potential_goal_hold_seconds`, `potential_goal_radius`, `potential_unseen_bonus`; `potential_snapshot()` and `field_context` with exact spec shapes. Spatial goal and metadata owned by env. Only field values are local; timestamp history remains available. Scene init remains seeded and validated. Radar actual pan follows explicit commands at finite rate.

- [x] Test P(0)=-1, P(T)=1, exponential monotonic recovery, cloud -2 override outside ROI, timestamp re-entry.
- [x] Test action[3] and finite pan, unit speed, legacy obs unchanged, local cache bounded by 33².
- [x] Implement field query and env integration; run new tests plus dynamics/environment regressions.
- [x] Reject certified unavoidable initial flight collisions, retain deterministic rejection metadata and the original failure evidence; verify near-cloud heading-away controls remain accepted.

### Task 2: Field experts

Files: src/expert_field.py (new), tests/test_field_expert.py (new). Root selects these classes in CLI, leaving old experts intact.

Consumes exact Task 1 obs dictionaries plus Config. Produces `FieldCoverageExpert(config).reset()/act(obs)` and `FieldTrackingExpert(config).reset()/act(obs)`, returning float32[3]. Use current obs only, no env access. Read local preference and cloud masks for motion and radar choice. Existing SafetyGuidance is reused for physical hard forecast, not hidden automatic expert fallback in learned policies.

- [x] Test changes of local field change decisions, radar aim sign/frame, obstacle action safety, deterministic replay and common obs contract.
- [x] Implement public-goal coverage guidance, field-aware avoidance/ROI penalty and explicit finite-pan target aiming.
- [x] Run seeded experts and inspect failure records; repair demonstrated failure causes.

### Task 3: Versioned network and imitation pipeline

Files: src/policy_field.py (new), src/policy_tracking.py (loader only), src/train.py, src/datasets.py; tests/test_field_training.py (new).

Consumes spec/Task 1 shapes. Produces `FieldPolicy(task,hidden_size=128)` separate weights, CNN+MLP+GRU -> normalized XY + tanh pan, `FieldNeuralPolicy`, factory/loader/dataset support with explicit obs-field-v2 checkpoint metadata. Existing train argument `model_config={'potential_field':True}` selects v2; no duplicated trainer. Export loader works via existing load_policy API.

- [x] Test mixed protocol rejection, padding/GRU continuity, three-action loss and gradients through field and pan head, checkpoint roundtrip.
- [x] Implement dynamic action dimension metrics denominator, field dataset loading, TF six relevant tags only.
- [x] Add optional shared permutation-invariant exact cloud encoder; preserve old slots checkpoints; fresh cross-architecture training pins historical scene splits without copying weights/normalization.
- [ ] Validate small separate datasets/train and held-out pure-policy closed loops; report failures honestly.

### Task 4: Entrypoint, replay and visualization integration

Files: main.py, run.sh, src/rollouts.py, src/collect_dagger.py, src/expert_validation.py, src/rendering.py, visual/map.js, visual/layers.js, visual/index.html; tests accordingly.

- [x] Wire potential toggle and every new Config field to run.sh config and CLI selector; choose expert/models by explicit schema.
- [x] Replace action dimension assumptions in rollout and validation; preserve original labels versus actual executed action, include actual pan.
- [x] Add signed local field overlay with fixed color range and controls, historical replay compatibility and MP4.
- [ ] Run complete regressions, field expert scenario suite, dataset/train/held-out validation, browser checks; write reproducible evidence and docs, commit branch without merging main.

## Interface audit and rulings

| Tasks | Shared interface | Decision |
|---|---|---|
| 1/2 | potential and field_context shapes | Exact spec indices; field experts use only public obs. |
| 1/3 | action[3] and protocol metadata | Explicit obs-field-v2; no shape guessing or v1 cache reuse. |
| 2/4 | expert factory | Root edits main selectors; experts are separate new classes. |
| 3/4 | create_policy and training config | potential_field bool selects new model; existing flags filtered for field class. |
| 1/4 | new Config fields | Environment owner lists defaults; root exposes and audits all run.sh entries. |
| all | feasibility vs perfect performance | Record precise tested scope and failures, do not claim universal guarantee. |

Ruling: work in the current directory on the user-requested new branch, preserving local outputs/services; no extra worktree or additional permission loop. User explicitly requested implementation of this architecture and confirmed the normalized exponential endpoint and timestamp-history definition.

## Evidence and remaining acceptance work

The progress-aware goal timeout passed the frozen 24-case field expert suite (16 coverage, 8 tracking), without excluded cases. Two additional six-hour coverage runs completed full revisits, while revealing revisit tails longer than the configured field decay period; selection alternatives are being compared with the field curve unchanged.

Eight canonical expert episodes per task were collected and JSON/NPZ parity checked. Early low-MSE models failed pure-policy closed loops; their weights, failure replays and metrics were retained. Body-frame inputs, explicit radar reference coordinates and validation-only deployment selection address measured failure causes. Final network acceptance remains separate from expert acceptance; the final validation report records the actual outcome.

Coverage epoch31 passed its two validation layouts and four frozen 32000-series tests but collided with a cloud at37s in the exact7100 demonstration. Deployment aliases were withdrawn and its failure artifacts retained. Repair now explicitly adds7100 to TRAIN, uses existing public cloud geometry through the shared encoder and collects expert labels on student-visited TRAIN states. New final seeds32100–32103 were declared before repair training; they must remain untouched until selection is frozen. The original validation layouts12000/12005 remain VAL across fresh architecture training. Full242 tests pass, but the network closed-loop acceptance items remain open.

The shared-cloud repair355211 passed both fixed VAL cases, TRAIN7100 and all16 known initialization regressions, but consumed32100-series final tests yielded3/4:32103 collided at5481.1s. No coverage publication followed. Exact action replay identified near-±pi control commands with opposite saturated-bank requests. An opt-in execution-aware excess-BCE training objective now passes250 full tests; inference and physical source402 remain unchanged. The next run is predeclared as fixed last10-only from355211, same63 data/splits/normalization, lambda0.05; raw-MSE best remains a diagnostic artifact. Known21 cases include the original16, consumed four tests and TRAIN7100; fresh32200–32203 stay untouched until that new candidate passes. No equal-length lambda0 continuation control has been run, so improvements cannot be attributed to the auxiliary term alone. These training and verification preparations do not close the network acceptance items.
