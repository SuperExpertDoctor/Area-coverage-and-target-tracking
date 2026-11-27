# 正式采集、训练与前端录像实施记录

本次用户授权提交并合并main、清理outputs/tf-logs、重新执行run.sh、每任务1000成功回合、8:2训练/验证、持续监督、最新权重双任务MP4及GitHub发布。两个任务独立，另有未见场景闭环测试。

## 当前证据与验收口径

以下为清缓存前的开发验证，不能当成本轮1000回合正式验收。小规模真实run.sh端到端试跑已完成：tracking两条30秒示教、一轮训练、950000独立场景、最后权重原SHA冻结、真实浏览器2秒640×360 H.264 MP4，所有流程成功。极小数据首轮验证损失反而变大，证明流程不会伪造单调下降；正式训练将观察完整曲线和闭环结果。

正式覆盖主要运行模式为显式forecast物理约束，纯网络同种子独立测试并单列结果；跟踪为纯网络。影片只使用本轮last.pt；不能使用best/旧模型或专家替换失败。所有失败计数和回放保留。源码SHA、数据身份、最终测试种子及模型SHA预先校验。

## 清理范围

用户要求清除旧缓存；实际递归删除被环境自动执行审查拒绝，旧outputs和tf-logs已隔离到被Git忽略的`.run-archive/20261128-preformal/`，随后新建空目录运行。旧产物未物理删除，正式流程不读取隔离目录。Git历史保留旧验证说明，克隆仓库不包含本地产物。正式结果将在运行后追加。

以下分模块开发记录为当时的历史快照，其中“未开始正式采集”等措辞不代表当前状态。当前阶段以`outputs/formal/formal-status.json`为准，完整正式结果以`formal-report.json`为准；开发通过率不得当成本轮1000回合最新模型的任务通过率。


---

# Formal collection implementation and measured evidence

2026-11-28; collection implementation development snapshot. Measurements are preserved here independently of the local cache lifecycle.

## API and behavior

`collect_expert_episodes(task, config, output_dir, *, episodes, seed, workers, attempt_multiplier, replay_every, progress_every, server=None, resume=True, require_task_success=False)` returns a JSON-safe authoritative manifest including `manifest_path`, `accepted`, `rejected`, `unused`, `attempts`, accepted NPZ `paths`, target, status and metrics.

- Acceptance uses the first N valid successes in an increasing contiguous seed prefix. Completion order never selects scenes. The submission/result window is at most `workers`; late speculative successes are explicit `unused` records.
- Initial `accepted-manifest.json` has `complete=false`; each completed attempt is atomically persisted. Worker-owned small sidecars recover results completed before a parent manifest update. Only exactly N finalized valid successes allow `complete=true`. Insufficient valid episodes raises with the manifest path and retains `complete=false`.
- Identity binds task, complete Config, source SHA, starting seed, requested count, attempt budget, replay cadence and task-success filter. Worker/progress counts may change on resume. Resume validates canonical JSON/NPZ artifact hashes. Changed identity or damaged/missing recorded artifacts fails explicitly; no silent recollection or cache mixing.
- Workers use the unchanged existing task environments and expert classes and share the exact Config/observation protocol. They write canonical episodes directly, return small summaries only, import no torch, and never own VisualServer. Bounded nonblocking display events carry collection attempt/seed; parent forwards only one active attempt scene/frame stream, while worker progress remains identified.
- `run_episode(..., replay_every=1)` preserves the existing full-cadence numerical behavior. Sparse replay stores initial, cadence and terminal frames plus explicit `frame_action_indices`. Every T+1 observation, T expert label and T actual executed action remains in canonical JSON and derived NPZ. Unrecorded steps use the identical underlying physical velocity without constructing full field/grid renders.
- Physical/visibility rejection, known planning failure, scene initialization failure and unexpected worker error remain separate evidence. Known failed experts never produce accepted caches. With `require_task_success=True`, physically safe but mission-incomplete cases retain truthful expert-valid JSON/NPZ and are excluded by manifest reason (`coverage_incomplete`, `tracking_incomplete`, `episode_duration_incomplete`). Full configured duration is mandatory. Legacy default remains False.

## Real 300-second field-v2 batches

Each row requests exactly eight accepted physically valid episodes, replay cadence60; coverage starts seed2000, tracking seed12000. The task-success filter is False for this throughput-only short benchmark. CPU load from other existing project processes was present. Startup and late speculative completion are included.

| Task | Workers | Wall s | Accepted | Rejected | Unused valid | Max worker peak MB | Sum per-worker peak upper bound MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coverage | 1 | 35.36 | 8 | 0 | 0 | 94.2 | 94.2 |
| coverage | 4 | 15.46 | 8 | 0 | 2 | 93.7 | 372.5 |
| coverage | 8 | 11.40 | 8 | 0 | 1 | 93.5 | 742.5 |
| tracking | 1 | 17.40 | 8 | 0 | 0 | 96.6 | 96.6 |
| tracking | 4 | 7.43 | 8 | 0 | 2 | 95.0 | 379.7 |
| tracking | 8 | 6.07 | 8 | 0 | 1 | 95.6 | 753.7 |

Coverage mission completion was0/8 in every300s batch; these short benchmarks do not establish mission success. Tracking was8/8 fully visible. Canonical+cache accepted storage for eight300s scenes: coverage30.24MB; tracking45.11MB. Eight workers was fastest in this sample.

## Real full-resolution versus sparse parity

120-second runs compare default cadence1 versus cadence60 with identical task/Config/seed. Every decoded raw observation tensor, expert command, actual executed action, scene and physical summary field was bitexact; only timing/display frames differ. Reconstructed NPZ parity is covered by tests.

| Task | Seed | Steps | Full frames | Sparse frames | Full JSON MB | Sparse JSON MB | Full wall s | Sparse wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coverage | 40 | 120 | 121 | 3 | 4.999 | 3.235 | 3.09 | 2.21 |
| tracking | 60 | 120 | 121 | 3 | 6.319 | 3.689 | 2.26 | 1.84 |

Scene route/config data creates a fixed JSON cost; sparse frames reduce repeated render payloads without removing training data.

## Full-duration mission-success pilot

Two workers; task-success filter True; cadence60; default Config with potential field enabled; two attempts per task, no rejected/unused scenes.

| Task | Seeds | Duration per episode s | Batch wall s | Accepted/rejected | JSON MB total | NPZ MB total | Max worker peak MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coverage | 2000,2001 | 10800 | 122.57 | 2/0 | 61.39 | 34.79 | 725.2 |
| tracking | 12000,12001 | 1800 | 6.10 | 2/0 | 24.50 | 12.51 | 177.2 |

Coverage seeds2000/2001 both reached coverage_ratio=1 and rescan_ratio=1, with no collision/out-of-bounds and first full coverage at4593.1/4776.7s. Tracking seeds12000/12001 stayed fully visible for1800s with no first loss or physical failure. This two-scene pilot does not establish the rejection rate of the forthcoming1000-scene runs.

## Formal-run recommendations and limitations

- Use8 workers, collection replay cadence60 and exposed progress cadence300, with require_task_success=True for formal data. Keep ordinary evaluation/recording rollouts at full default cadence.
- Measured full-duration canonical JSON+NPZ averages48.09MB per coverage episode and18.51MB per tracking episode: roughly66.6GB for1000 accepted episodes per task, before rejected/unused episodes, manifests, model files and logs. Every retained rejected safe-but-incomplete episode also consumes storage; monitor free disk and actual rejection rates.
- Full coverage worker peak was725MB, giving a rough5.8GB eight-worker peak upper estimate plus parent/server/OS and other stages. Short-run memory figures must not be substituted for full10800s memory.
- Two-worker full coverage throughput was2 accepted/122.57s:1000 scenes would take about17h at that measured two-worker rate. Ideal fourfold scaling to8 workers suggests4.3h, but CPU contention, scene variation, rejected attempts and slower canonical serialization can increase this. Use actual formal progress rather than treating this estimate as a guarantee. Tracking is much faster.
- Keep all hyperparameters in run.sh/main.py (root-owned). No source in config, environments, dynamics, fields or experts was modified. No outputs cleaned, services stopped, models reused or commits made by this task.

## Validation

Final scoped verification: `python -m pytest tests/test_collection.py tests/test_rollouts.py -q --basetemp=outputs/cache/pytest-formal-collection` ->43 passed. Meaningful cases include deliberate out-of-order completion, ordered rejection, bounded window, interrupted/resumed manifests, sidecar recovery, identity drift, artifact tampering, process failures, initialization/planning/unexpected-error evidence, queue saturation, real spawned-worker scene/frame context and sparse full-label preservation.

An earlier extended check exposed two CLI failures from a concurrently edited top-level visual_server relative import. The recording owner fixed that import. Final extended command `python -m pytest tests/test_collection.py tests/test_rollouts.py tests/test_field_integration.py -q --basetemp=outputs/cache/pytest-formal-collection` passed46 tests in13.63s; output is recorded in collection-tests.log. `py_compile` and scoped `git diff --check` also passed (Git only reports pre-existing line-ending normalization advice).

Final current source fingerprint: `07bc1fafcfae0bff4c2b3fbcfafe4a531d29387a7a062d23f2f882843c441e72`. Throughput/full-duration pilots preceded the last source-identity guard and cadence-input validation additions; those additions change validation only and do not change Config, expert, physics, tensors or stored replay content.

Historical raw evidence was generated under outputs/formal-development: collection-benchmark.log; collection-benchmark/{batch-results,parity-results}.json; collection-full-duration.log; collection-full-duration-results.json; per-task accepted-manifest.json and canonical episode/cache artifacts. This summary preserves those measurements; raw artifacts are local and excluded from Git.


---

# Task2 frontend recording implementation and verification

Scope: visual/recording.js, visual/app.js, visual/index.html, visual/styles.css, src/visual_server.py, src/browser_recording.py, tests/test_recording.py and recording cases in tests/test_visual_frontend.py. No dynamics/environment/training/policy changes. No requirements changes, service takeovers, cleanup or commits.

## Result

The existing simulation console now has start/stop/status/error/download controls. The actual rendered mission canvas and its existing ocean backdrop are copied into the configured recording canvas; task, actual replay time, model SHA/path, control mode, actual metrics and failed-task status are burned into the video. The recording settings shown by the UI come only from /api/recording-config. Runtime MediaRecorder MIME detection prefers native MP4 and falls back to supported WebM. POST /api/recordings decodes/transcodes to H.264/yuv420p MP4 using the existing imageio_ffmpeg FFmpeg; it never renames WebM.

The same MissionRecorder serves user buttons and window.maritimeRecording.recordReplay. /?replay=/outputs/<encoded-relative-path> and the same-origin bridge load real replay JSON. record_policy_replay(server_url,replay_path,output_dir,*,fps,width,height,duration_seconds,headless=True) invokes that bridge in an actual Chrome/Edge/Chromium browser and then decodes the resulting MP4, checks dimensions/fps/duration and proves the real replay endpoint was reached. output_dir means the VisualServer output root, and the result path is under its recordings child. Policy-only guard rejects expert and DAgger training-mixture substitution. Failed and early-terminated replays remain failed; no future flight is synthesized. Full replay time is compressed into the requested video duration, including exact first/final real frames. Manual early stop retains its actual partial endpoint.

VisualServer accepts recording_config as an optional keyword, exposes GET /api/recording-config and bounded POST /api/recordings. Settings are recording_fps, recording_width, recording_height, recording_duration_seconds, recording_bitrate, recording_max_bytes, recording_max_seconds. The defaults are 24,1440,900,30,8000000,268435456,600. Config requires finite positive values/even H.264 dimensions. Raw uploads require Content-Length and video/mp4 or video/webm; byte count, metadata bytes, finite JSON, duration, concurrent conversion, socket read, FFmpeg wall time and resulting output size are bounded. Output names are server-generated and directory/path resolution is checked. Foreign-origin UI uploads are rejected. A corrupt source, conversion timeout/failure or oversize video returns structured error JSON and leaves no fake MP4.

GET /api/run-status reads formal-status.json (missing/malformed returns{}). UI also handles SSE run_status messages and shows the actual phase/collection counts/epoch/loss. It does not move vehicles during collection/training status updates. Package and existing top-level src imports both work.

## Browser environment and documentation

Browser plugin not available, so regular installed Node Playwright and installed Chrome were used. Runtime discovery prefers PLAYWRIGHT_MODULE_PATH, normal project/portable module resolution, then dynamically discovers existing npm-cache installations without hardcoding a cache hash. No browser/package download occurred. Install alternative: npm install playwright; use installed Chrome/Edge or npx playwright install chromium. Node.js is required for the automation helper; the frontend manual facility only requires a supported browser.

Primary documentation checked:
- https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/captureStream — canvas real-time stream, frame-rate behavior, origin-clean requirement.
- https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder/isTypeSupported_static — runtime MIME feature detection.
- https://developer.mozilla.org/en-US/docs/Web/API/MediaRecorder — recording/chunk/stop behavior.
- https://playwright.dev/python/docs/browsers — support for installed Chrome/Edge channels, current headless behavior.
- https://playwright.dev/docs/api/class-browsertype — installed-channel browser launch options.

## Automated checks

Command: python -m pytest tests/test_recording.py tests/test_visual_server.py tests/test_visual_frontend.py tests/test_main.py -q --basetemp outputs/cache/pytest-formal-recording-review

Result:50passed,1skipped in19.39s. Includes backend actual WebM-to-MP4 decode, MP4 ftyp/header verification, full FFmpeg decode, served-download bytes, traversal-resistant generated names, oversize/MIME/origin/metadata/corrupt/timeout errors, run-status API, actual-browser shared UI capture, retained model SHA/control mode/failure, timeline endpoints, and substitution rejection. Existing visual-server/frontend/main cases are included. Windows symlink test skips when developer-mode/admin symlink privilege is unavailable; existing traversal tests and resolved-parent enforcement still run.

## Actual browser QA

Flow under test: localhost app -> real replay load -> visible start recording -> configured-duration save/download, and mobile manual stop -> actual partial save; backend conversion failure -> visible error.

| Check | Desktop 1440x900 | Mobile 390x844 |
| --- | --- | --- |
| Correct page URL/title, meaningful UI | pass | pass |
| No framework overlay | pass | pass |
| JS/console errors | none | none |
| Horizontal overflow | none (1440px) | none (390px) |
| Start/stop/save/download interaction | pass | pass |
| Real H.264 MP4 decoded | 48frames /24FPS=2.000s | 4frames /24FPS=0.1667s early stop |
| Metadata actual endpoint | 10800s |1020s intentional early stop |

The no-replay start action shows a meaningful error. An injected422 conversion failure displays `录制失败：QA conversion failure`. Desktop/mobile screenshots were inspected. Recording first/last video frames were extracted and inspected: visible SIM0 and SIM10800, model/path/mode, actual coverage0%/70%, and FAILED time_limit remain visible.

Default-settings preflight: the public Python API captured a full 30-second video through the actual shared frontend facility at1440x900/24FPS. Full decode produced720frames, H.264 MP4,30.000seconds,4820331bytes, SHA256483b33f67dd74505f9d710d8266cf98b6875d444217a8943fa1b2c7bef0d8556. This uses a clearly identified development fixture replay, not a trained-model result. Main/formal-run must call the API on each newly evaluated last.pt replay after training; no claim of latest trained-model success is made by this development report.

## Evidence files

- outputs/formal-development/recording-qa-results.json
- outputs/formal-development/recording-default30-result.json
- outputs/formal-development/recording-desktop.png
- outputs/formal-development/recording-mobile.png
- outputs/formal-development/recording-error.png
- outputs/formal-development/recording-video-first.png
- outputs/formal-development/recording-video-last.png
- outputs/cache/formal-recording-qa/recordings/20261127T202215-f888beed3f4a.mp4 (default30s development preflight)

Artifacts above are intentionally in ignored outputs and will be removed by the root agent's authorized fresh-run cleanup. Preserve this report's summary before cleanup.

## Remaining limits

Chrome was exercised; Firefox/Safari/WebKit were not. Actual WebM conversion was tested through FFmpeg-generated WebM and actual browser capture used Chrome's runtime-selected supported format. The Windows symlink-specific test needs elevated/developer-mode privilege. Formal latest-weight full-duration task outcomes and their post-training MP4s are owned by root's later formal run, not this fixture verification.

---

# Task3: training scale and observability evidence

Date: 2026-11-28 (Asia/Shanghai). Work is restricted to `src/train.py` and
`tests/test_training_progress.py`. `src/datasets.py` did not need modification.
All profiling uses old datasets with newly initialized scratch models. No old
weights or normalization were loaded; no formal 1000-episode run was started.

## Measured hardware and throughput

The machine exposes an RTX 5060 (8151 MiB), PyTorch 2.7.1+cu128 and 16 logical
CPU threads. About 2.9 GiB of GPU memory was already in use by other applications.
Measurements use float32, hidden size 128, the existing field CNN/GRU, body field
coordinates, the shared cloud encoder, 4 Torch CPU threads and ordinary Adam.

| Current-batch episodes | TBPTT sequence | NPZ load threads | Valid action steps | Train seconds | NPZ load seconds | Valid steps/s | CUDA allocated/reserved MiB | Process memory GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 (8 total across two batches) | 64 | 0 | 57600 | 6.60 | 1.90 | 8727 | 78 / 98 | 2.71 peak |
| 8 | 128 | 0 | 57600 | 3.51 | 2.03 | 16415 | 314 / 506 | 2.70 peak |
| 8 | 128 | 4 | 57600 | 2.49 | 0.66 | 23133 | 314 / 486 | 2.65 sampled |
| 16 | 128 | 4 | 115200 | 3.60 | 1.46 | 32028 | 599 / 968 | 3.65 sampled |
| 32 | 256 | 0 | 230400 | 10.41 | 7.36 | 22128 | 2326 / 3670 | 5.67 sampled |
| 32 | 256 | 4 | 230400 | 6.13 | 2.93 | 37569 | 2326 / 3670 | 5.54 sampled |

The 16- and 32-episode benchmark batches repeat the eight available full-length
7200-step coverage paths solely to measure bounded batch loading and GPU work;
they are not new demonstrations or a claimed independent training dataset.
The baseline tracking profile processed eight 1800-step episodes at batch 8,
sequence 128 in 0.94 seconds (15254 valid steps/s, 0.56 seconds NPZ loading).

Recommend `batch_size=16`, `sequence_length=128`, `load_workers=4`,
`num_threads=4`, `progress_every=50`, `progress_seconds=30`. The larger 32/256
batch gives only about 17% more throughput while reserving approximately 3.7 GiB
of CUDA memory; 16/128 leaves more space for the real frontend and desktop.
These settings change the usual optimizer batch/chunk schedule when selected;
they are explicit parameters, not hidden alterations to the loss.

For 1000 independent 10800-step coverage episodes, the uniform-length benchmark
suggests roughly 5–6 minutes per train/validation epoch; 1800-step tracking is
roughly another minute. Twenty epochs of both tasks are approximately 2–3 hours
of training, excluding expert collection and final closed-loop recording.
This is an estimate, not a completed formal-run measurement. Unequal lengths
increase padded work; hardware load and held-out proportions affect elapsed time.
Reestimate from the first actual formal epoch. Epoch count must be judged against
validation and independent closed-loop behavior, not throughput alone.

Four bounded loading threads address the measured decompression bottleneck.
Only the current episode batch is materialized; previous slices and their whole
backing episode storage are explicitly released before the next batch loads.
No full-RAM dataset load, duplicated all-map array, quantization, mmap cache or
large disk cache was added. At 1000x10800 coverage steps, an uncompressed field
cache alone would approach 180 GiB, making it a poor default on this disk.

## API and published progress

New optional `train_policy` keyword arguments:

- `progress_every=50`: publish after this many completed TBPTT chunks.
- `progress_seconds=30.`: publish when this amount of phase time passes, even
  if the chunk count has not reached its threshold.
- `load_workers=0`: zero retains sequential NPZ loading; positive values load
  only the current episode batch with at most that many threads.
- `resume_checkpoint=None`: explicit completed-epoch continuation from a new
  `last.pt`; distinct from optimizer-reset `initial_checkpoint` warm starts.

`output_dir/training-status.json` is atomically replaced during preparation,
epoch-zero validation/train baselines, every batch load, bounded chunk/time
progress, completed epochs, completion, and failures. Its schema includes:

```json
{
  "task": "coverage",
  "pid": 1234,
  "state": "running",
  "phase": "train",
  "operation": "chunks",
  "epoch": 1,
  "epochs": 20,
  "processed_steps": 16000,
  "chunks": 50,
  "episodes_completed": 0,
  "episodes_total": 800,
  "metrics": {"loss": 0.1, "action_mse": 0.033, "action_mae": 0.12, "steps": 16000},
  "train_metrics": null,
  "validation_metrics": null,
  "gradient_norm": 0.5,
  "elapsed_seconds": 2.0,
  "data_load_seconds": 0.3,
  "run_elapsed_seconds": 80.0,
  "updated_at": "2026-11-28T00:00:00+00:00",
  "validation_available": true
}
```

The partial `metrics` describe the current phase. `train_metrics` and
`validation_metrics` contain completed metrics, starting with epoch zero.
`gradient_norm` is the latest pre-clipping gradient norm for a training phase;
it is null for evaluation. No extra TensorBoard tags were introduced.
Supported phases are `preparing`, `normalization`, `initial_validation`,
`initial_train`, `train`, `validation`, `selection`, `epoch_complete`, `complete`.
Failures set `state=failed` with an `error` string and raise the original error.

`history.json` and the existing `training_history.json` preserve the result
schema and are replaced after each completed epoch, including a completed
epoch-zero snapshot. They are not withheld until the end of a long run.

## Atomic persistence and continuation contract

JSON, best/last checkpoints and the task-named export use same-directory
temporary files, flush/fsync and atomic replacement. Failed writes leave the
previous valid artifact unchanged and remove their own temporary file. An actual
Windows `WinError 5` was reproduced by keeping the destination open for reading;
replacement now retries only `PermissionError`, with a bounded 1.27-second
backoff, retaining the prior snapshot throughout. Permanent errors still raise.

Every `last.pt`, including epoch zero, stores model state, Adam state, Python,
NumPy, Torch CPU/CUDA RNG states, the local episode-shuffle generator state,
history, initial metrics, best selection metadata, the exact scene split and
bounded-buffer SHA256 fingerprints of all accepted dataset files. It embeds
one SHA-verified best artifact, so a crash after publishing a newer best but
before publishing last cannot strand the last checkpoint with an inconsistent
best. This adds about one small checkpoint to the last file, not dataset-sized
memory or storage. The measured last files are approximately 11.2 MB.

Continuation rejects changed dataset bytes/paths/scene metadata, task/protocol,
architecture, numerical training settings, PyTorch/NumPy/CUDA/cuDNN environment,
and runtime precision/determinism flags. `epochs` is a new total target and may
increase; loading threads and status cadence may change. It resumes only at a
completed epoch boundary and does not pretend that an interrupted GRU chunk was
saved. TensorBoard uses `purge_step=restored_epoch+1` and never emits another
epoch-zero point during continuation.

CPU continuation and CUDA continuation with deterministic kernels were checked
bitwise for all model/Adam tensors, all RNG states, scene split, metric history
and best epoch. The CUDA proof uses deterministic PyTorch/cuDNN plus
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. Ordinary CUDA kernels
showed up to 3.7e-9 variation in a few weights despite the same restored state;
the API documentation explicitly distinguishes exact state restoration from
bitwise reproducibility under nondeterministic kernels. No runtime precision or
determinism setting is changed silently by the trainer.
See the primary [PyTorch reproducibility documentation](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md).

Nonfinite policy predictions, loss and gradient norms fail before publishing
new weights or scalar metrics. Tests show that a failed first update leaves the
epoch-zero last checkpoint valid and the status truthfully failed.

## End-to-end and test evidence

The final production trainer was run with fresh weights against representative
old field datasets and separate scratch output/log directories:

| Task | Source episodes | Train / validation episodes | Train / validation valid steps | Epoch zero + epoch one elapsed |
|---|---:|---:|---:|---:|
| Coverage | 82 | 72 / 10 | 77362 / 29405 | 18.29 s |
| Tracking | 12 | 8 / 4 | 14400 / 7200 | 1.94 s |

These old files include repeated scene groups and short DAgger prefixes, so the
episode ratio reflects disjoint layout grouping rather than an exact episode
count ratio. Their variable lengths create padded work and are not the uniform
long-episode scaling benchmark. The formal run will use new data and new splits.

Both end-to-end runs produced valid epoch-one checkpoints with 32 populated Adam
state entries. TensorBoard event parsing found exactly these six tags, each at
steps 0 and 1: `Train/Loss`, `Train/Action_MSE`, `Train/Action_MAE`,
`Validation/Loss`, `Validation/Action_MSE`, `Validation/Action_MAE`.
The resumed CPU/CUDA proof separately checked exactly steps 0,1,2,3.

Verification command:

```powershell
python -m pytest tests/test_training_progress.py tests/test_training.py tests/test_field_training.py --basetemp=outputs/cache/pytest-formal-training -q
```

Latest result: **92 passed in 20.61 seconds**. `git diff --check -- src/train.py`
also passed. New tests exercise atomic interrupted writes, transient/permanent
Windows locks, interval/time progress, byte-exact parallel-loader equivalence,
bounded episode backing storage, incremental epoch-zero/history status,
CPU/CUDA resume with all optimizer/RNG/history state, TensorBoard deduplication,
changed-data/config/runtime rejection, interrupted checkpoint-pair recovery,
preparation failures and immediate finite-value guards. Existing loss, padding,
normalization, historical split and field-specific tests remain passing.

Raw evidence lives under `outputs/formal-development/` (`training-profile.json`,
`loading-profile.json`, `trainer-e2e-profile.json`, profiling scripts and red/green
logs); scratch event logs are under `tf-logs/formal-development/training-profile`.
The parent should copy the relevant measured summary into tracked documentation
before the authorized old-output/log cleanup.


## 合并前最终验证

全量回归：345 passed,1 skipped（Windows symlink权限），148.10秒。之后入口增加显式推理线程参数、独立测试失败保留及采集接口接通，相关32项测试通过，16.97秒。独立审查确认源码/种子隔离、真实加载SHA、last.pt不回退、纯网络/约束模式分开统计；实际8794服务收到coverage专家6480–9600秒实时帧，桌面正确展示任务/时钟/阶段/录屏设置，无JS错误或水平溢出。

原子状态写入对Windows临时读锁有界重试；正式运行前设置NumPy/BLAS线程预算，保持所有物理积分精度。上述检查属于开发验证，正式1000+1000采集尚待清缓存后启动。


完整时长小批量流程使用过940000/950000开发场景，其中两回合/一轮覆盖纯网络在667秒碰云，此失败没有隐藏或替换。正式最终测试在运行前改用尚未查看的960000–960007和970000–970007，后续不据测试结果挑选best/last。


## 完整演示暴露的候选搜索缺陷

两回合、一轮训练的覆盖模型在7100演示10392秒完成100%首扫/重扫后，旧forecast层的通用几何修正抛出NoFeasibleAction，系统正确停止并保存FAILED影片；跟踪示范1800秒全程可见并保存MP4。独立145角度搜索确认当时仍有6个安全方向（0.698–0.916弧度），最近云安全净距704.24m，因此问题是候选集不足，而非已证明物理无解。

新增真实两云状态回归测试，先失败再修复：在通用修正失败时额外预测现有配置的航向偏置和上一条安全指令，每个候选都从当前状态检查；原始安全动作bitexact透传，pan保持原值，没有任务专家回退。增加版本forecast-v2-candidate-search和candidate_repairs审计。47项相关测试通过；独立审查41项通过，并确认上一条指令也须重新验证。该有限候选补救仍不构成全局安全证明。

另修复入口日志编码的兼容问题：本机Conda有GBK编码.pth，强制PYTHONUTF8会在启动阶段失败；入口仅设置标准输入输出UTF-8，不改变site初始化编码，真实bash/Python --help启动通过。所有改动在正式清缓存/1000回合采集前完成。


修复后同一75dfba95…最后权重、同一7100初始化完整重跑：10800秒安全结束，首扫/重扫均1.0，首扫5196秒；238次公开约束干预，其中12次候选补救，failed_actions=0。真实前端再录制640×360/24FPS/48帧/2秒H.264 MP4，解码校验成功，记录0→10800秒和实际模型SHA。此为小批量开发验证，不计入正式1000回合结果。最新全量测试347 passed,1 skipped，145.25秒。


## main合并与正式重启准备

代码于2df4d9c合并main并推送GitHub，远端SHA复核一致；合并结果与已审查分支内容相同，43项关键测试通过、1项Windows权限跳过。

实际清理时，环境自动执行审查以“blocked by policy”拒绝动态校验和固定绝对路径两种递归删除命令。采用可恢复方案：校验旧目录与隔离目录均在本workspace且无junction/symlink，停止经验证的旧服务/浏览器守护进程，将旧outputs与tf-logs移入.run-archive/20261128-preformal/，然后重新创建两个空目录。旧产物仍在隔离目录，不算已物理删除；新流程不读取隔离目录，归档也加入.gitignore。所有训练从新生成数据和初始化权重开始。


正式启动审计发现协议序列化缺陷：Config中的tuple经JSON持久化后变成list，导致完全相同的协议误判变化，源SHA始终一致。首批尚无已完成示教时停止，旧启动输出隔离到outputs/cache/protocol-preflight-20261128，修复为统一JSON原生表示；新增JSONroundtrip及真实持久化协议续跑入口测试。相关36项和续跑入口8项通过。随后重启实际run.sh，从0采集，未将首次启动的任何轨迹纳入正式清单。

## 正式监督发现的云簇局部环路

40af8ab的覆盖采集在173个已接受回合、1个未完成回合的监督快照后中止。seed12153安全运行满10800秒，但3510个海面格中仍缺6格，覆盖率0.9982905983。原动作逐步回放与180个原记录帧的状态/覆盖统计误差均为0，排除了并行采集和展示的伪影。末段6386秒反复服务同一目标，约486秒周期的轨迹返回误差最大仅1.495米；t4415之后距该目标至少3054.58米，超出3000米雷达范围。

该场景存在符合原硬约束的扫描位置：目标北侧2500米、航向向东、原右侧雷达pan0可同时扫描六格；对应R=1000米整圆对所有膨胀雷云边界的最小净距665.33米。因此不能将失败归咎于不可扫描场景。过大的1300米软避障缓冲与云簇形成局部外绕环路，超时后重选同一目标又放大了滞留。

单变量诊断把场覆盖软缓冲改为200米，3692.7秒即完成首扫；但在7569秒又暴露候选遗漏：原10个指令全部不通过30秒预测。独立187方向审查发现35个方向通过完全相同的5秒步长保守预测，这35个也全部通过0.1秒物理步长复核。因此仅缩小软缓冲不足以验收完整任务。

该轮最小修复增加`field_coverage_obstacle_buffer=200`及入口显式参数/非负有限校验；只影响场覆盖专家的软引导，硬膨胀量、转弯半径、滚转、雷达范围、预测时长和圆弧保守界不变。SafetyGuidance复用原预测计算，原候选有可行解时保持原选择；原集合全失败时才搜索当前航向和原始请求航向两参考上的`field_heading_offsets`与`expert_safety_heading_offsets`并集，每条都独立重预测。该公共补充也可救活跟踪原本的失败分支，不声称所有跟踪状态的行为完全不变。

真实7569秒状态与8000秒稳定环路保存为仓库测试夹具，不依赖outputs。TDD旧实现实际报规划失败或720秒后仍缺六格；修复后快速回归全部通过，原可行黄金指令逐位一致。全量测试358 passed、1 skipped，149.59秒。

修复后的生产专家使用同一原场景，从原始起点完整运行10800秒：首扫3692.7秒，最终覆盖率1、重扫率1，无机船碰撞/越界/规划异常；最小按决策步采样的云硬边界净距158.70米，额外候选启用4次。上述为该失败场景的修复证据，不是任意初始化保证。

被中止的正式目录及日志已隔离到`.run-archive/20261128-expert-local-trap/`，没有送入新一轮正式训练；新建outputs/formal后内容为0项，tf-logs中也尚无训练事件。完成额外开发场景回归与独立审查后，再以修复源码从0启动1000+1000个回合，而不混合修复前后的来源。

补充回归完成：覆盖7100/2000/2001/3002/4000均安全运行满10800秒、覆盖率1，重扫比例0.99447–1；跟踪2000/2001均满1800秒且全程可见。加上原12153，共6个覆盖、2个跟踪完整开发场景通过，始末源码SHA一致。这些开发种子未代替正式960000/970000系列未见测试。

独立审查78项通过；64组合中原本可行的61组指令与旧实现逐位一致。真实8000秒六格快照消融：保留新预测候选但恢复软缓冲1300米，720步后仍有六格未扫；软缓冲200米则33步内补全。真实7569秒返回指令也用未量化状态/原精确云参数独立复核，30秒物理轨迹最小硬净距248.14米。审查未发现阻塞；以上有限证据不构成任意状态的安全证明。

用户随后明确确认每任务1000回合（总计2000）、各800训练+200验证，另生成独立闭环测试；与已设定正式协议一致。

## 第二次候选集合缺口与全圆搜索

27e4b12已推送main并真实从空目录重启。第二轮正式采集在40个已接受回合、1个规划失败回合的监督快照后再次停止。seed12029已完成首扫和重扫，但8925秒时两级候选集合共32项全部拒绝。独立746候选审查有33个通过原30秒/5秒保守预测，且这33个全部通过0.1秒物理复核；安全窗相对当前航向约+0.447至+0.717弧度，现有+0.4和+0.8偏置恰好跨过该窗。上一执行指令也仍可行。因此需要补全搜索范围，不能将该回合仅归为物理不可行。

新增第三级全圆搜索，只有前两集合全失败才执行。数量由run.sh中的`expert_safety_fallback_heading_samples=145`指定，要求有限整数且至少8；以当前航向锚定均匀角网格，按与通用修正方向的角差依次执行原预测，返回首个可行方向。原分支已有可行解时保持原行为，硬约束不变。

首版绝对世界角网格会破坏机体系输入的旋转一致性。新增真实状态整体旋转行为测试后，绝对网格有3个旋转反例；修正为当前航向相对网格，53项相关检查通过。绝对网格的短暂开发并行运行已停止并标记superseded，不计入最终场景回归。

这轮40回合与日志已隔离到`.run-archive/20261128-heading-gap/`，formal目录重建后为0项，TF事件仍为0。随后使用固定相对网格源码，验证原12029/12153两个场景，以及覆盖10000–10031、跟踪11000–11007，共42个开发场景。正式最终测试960000/970000系列仍未用于任何调整。

相对网格版全量测试373 passed、1 skipped，246.26秒（同时有8个开发验证进程运行）。独立审查67项通过，额外37个整体旋转角最大命令分量误差5.13e-8。真实12029前32次原/并集候选全失败后，第138个网格方向通过，方向相对当前航向+0.454989弧度；未量化状态和原云参数的0.1秒物理复核，最小云硬边界净距26.45米。独立复核未发现阻塞，未修改产品源码或读取正式未见测试。

公共SafetyGuidance还被ForecastConstrainedPolicy使用，因此新全圆搜索也扩展其旧通用修正失败分支。安全的原网络动作仍原样透传，约束后动作仍由独立预测再检查；没有调用任务专家替代网络。有限145方向搜索不是任意可行角窗必被命中的证明。

固定相对网格版42/42开发场景全部通过，源SHA始末均为ad8ca666…，总wall778.95秒。34个覆盖场景都安全运行满10800秒、覆盖率1；首次全覆盖最小/均值/最大3109.3/5184.09/8397.8秒，重扫比例最小0.906015、均值0.995715。8个跟踪场景都满1800秒、可见率1且first_loss_time为空。全圆搜索仅在原12029的8925秒处触发一次；未将有限开发结果解释为任意初始化或T内重访保证。两个原场景的几何/初始状态SHA也与失败记录精确一致。
