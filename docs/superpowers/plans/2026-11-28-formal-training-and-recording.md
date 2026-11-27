# Formal Training and Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development for scoped implementation and independent review.

**Goal:** Run fresh large-scale two-task imitation learning with supervised progress and real frontend MP4 recording, then merge/publish the reviewed implementation.

**Architecture:** Reuse existing field-v2 environments/experts/CNN+GRU; add deterministic parallel collection, bounded storage, recording and a durable formal-run supervisor.

**Tech Stack:** Gymnasium, NumPy, PyTorch/CUDA, TensorBoard, native JavaScript Canvas/MediaRecorder, Playwright, FFmpeg.

**Spec:** docs/superpowers/specs/2026-11-28-formal-training-and-recording-design.md

## Global constraints

- User explicitly authorizes main merge, output/log cleanup and GitHub push. Do not add another approval loop for these actions.
- Keep all algorithm source in src, UI in visual, outputs/caches under outputs and event logs under tf-logs.
- Preserve common obs/action interfaces, field domain and kinematic equations. No hidden task-expert takeover or older-weight substitution in latest-model recordings.
- Use unique pytest basetemp per concurrent test process.
- Decide formal episode allocation and split semantics from pending user answers; implementation accepts both.

### Task1: Scalable deterministic collection

Files: new src/collection.py; src/rollouts.py; tests/test_collection.py and relevant rollout tests.

Interface: collect_expert_episodes(task, config, output_dir, *, episodes, seed, workers, attempt_multiplier, replay_every, progress_every, server=None, resume=True) returns JSON-safe accepted/rejected summaries and authoritative manifest path. The same Config and existing field expert classes are used. Source/config/seed identities prevent accidental reuse. Workers stream bounded scene/frame/progress events to a parent callback and return small summaries only.

- [ ] TDD deterministic seed ordering/rejections, exact count, resumed manifests/config identity, worker errors, full labels with sparse indexed frames.
- [ ] Implement bounded process collection and optional sparse frame recording, preserving default run_episode behavior.
- [ ] Benchmark a small real batch and report throughput/storage/RAM.

### Task2: Frontend recording and automatic capture

Files: visual/recording.js, visual/app.js/index.html/styles.css, src/visual_server.py, src/browser_recording.py; related tests.

Interface: VisualServer recording_config keyword and /api/recording-config; POST /api/recordings returns saved MP4 URL/metadata. record_policy_replay(server_url, replay_path, output_dir, *, fps, width, height, duration_seconds, headless=True) returns saved MP4 metadata. Main owns invocation after producing latest-weight replays. Read settings from recording_config, no hardcoded hidden hyperparameters.

- [ ] Verify current primary documentation for captureStream/MediaRecorder/browser automation.
- [ ] TDD upload bounds/path safety/FFmpeg failure and format authenticity; UI start/stop/status/errors and same-origin replay loading.
- [ ] Actual browser desktop/mobile capture, MP4 decode and timing/visible-mode proof.

### Task3: Large-run training and observability

Files: src/train.py, src/datasets.py if measured fixes needed, related tests.

- [ ] Profile current epoch I/O/GPU/memory on representative existing data before cleanup.
- [ ] Add incremental atomic training status/history and exact optimizer/RNG resume if needed; preserve existing loss/split/padding behavior.
- [ ] Keep six TensorBoard tags truthful; validation/test semantics explicit; report epoch throughput and recommended exposed settings.

### Task4: Entrypoint, supervisor and documentation

Files: main.py, run.sh, new src/formal_run.py/monitoring.py as needed, README.md.

- [ ] Wire all Task1/2/3 parameters through run.sh/main.py and independent two-task stages.
- [ ] Parent server and live stage progress; latest last.pt recordings for each task with SHA provenance and honest task outcome.
- [ ] Independent whole-diff review; full tests; commit and merge main; resolve/clear only authorized output/log directories.
- [ ] Start actual formal run; supervise collection/training/TF/log/browser; fix actual issues and rerun affected work.
- [ ] Verify latest-model task outcomes and MP4s; update actual README/results; commit/push main and verify remote.

Ruling: work in the current authorized feature branch/workspace and later merge into main as requested. Prior behavior/model failures remain historical evidence in Git documentation; the user now explicitly requests deleting the local cached artifacts. New runs must not reuse old data, normalization, weights or TF events.
