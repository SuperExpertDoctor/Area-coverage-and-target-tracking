# Experiment Artifacts Implementation Plan

> **For agentic workers:** Use delegated independent tasks for NPZ storage, batch training and evaluation; root integrates CLI, formal workflow and documentation.

**Goal:** Organize experiment datasets, weights, logs and 30-episode evaluations while serving compact NPZ trajectories as frontend JSON and logging each training/validation batch.

**Architecture:** A shared episode I/O layer owns NPZ/legacy conversion. Existing rollout, collection, viewer and recording consumers use it. Training separates weight and JSON log destinations and exposes validation batch sizing; evaluation orchestrates the existing policy rollout using frozen weights. CLI and formal execution choose all paths before work starts.

**Tech Stack:** Python, NumPy compressed NPZ, PyTorch/TensorBoard, standard-library HTTP, native JavaScript, pytest and Playwright CLI.

- [x] Add failing NPZ roundtrip/collection/HTTP tests; implement `src/episode_io.py`, adapt rollout and consumers, verify no full JSON duplicate and no pickle.
- [x] Add failing variable-length batch/TensorBoard tests; add validation batch size, batch callbacks/writers/counters and separate `log_dir`; preserve low-level legacy checkpoint compatibility.
- [x] Add failing 30-episode evaluation/partial-report/failure tests; implement `src/evaluation.py` around existing formal rollout.
- [x] Add CLI workflow tests for weights/logs/cache/eval paths, aliases and mandatory post-training evaluation; run them red, integrate main.py, formal_run.py and run.sh, then run green.
- [x] Apply follow-up requirement: reject existing named experiments in either parent before side effects, remove CLI resume options, keep serve from rewriting run configuration, and use distinct experiment names in staged examples.
- [x] Rewrite README around verified entry commands, four experiment directories, NPZ format, regenerable caches, batch semantics and independent evaluation success rates; update visual docs.
- [x] Run focused then broad regression, actual short CPU integration and browser NPZ replay/recording; inspect diff, preserve active original run and user documentation changes.
- [ ] Commit, attempt authorized remote push, verify remote state if successful, and report any remaining authentication block with exact local commit/worktree.

Verification evidence: real CPU tracking integration collected 4 accepted 4-second episodes, trained/validated one epoch (4 train batch events and 2 validation events), and archived 30 independent inference episodes. The actual browser loaded one NPZ and produced a decoded 320x240, 10 FPS, 10-frame MP4 with matching source SHA under logs/recordings. These short episodes verify integration only. Review fixes preserve prior NPZ until atomic replacement, prevent stale archives from being attributed to failed new attempts, and exclude historical fixed scene groups. The original algorithm process and original checkout remain untouched.

Final broad regression: 507 passed, 1 skipped (Windows symlink privileges); final CLI/formal checks including two added race/link cases: 82 passed. JavaScript syntax, README local links and git diff whitespace checks passed.
