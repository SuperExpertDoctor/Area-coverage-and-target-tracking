# Maritime visualizer

No frontend build or npm install is required. Start the project CLI with its
visual option, or run `python src/visual_server.py --port 8787`, then open
`http://127.0.0.1:8787/`.

The static frontend consists of `index.html`, `styles.css`, `app.js` (controls,
SSE and measured metrics), `map.js` (Cartesian map and loaded 2D model assets),
`layers.js` (sparse temporal visits, clipped trails and physical model sizing),
`field.js` (masked preference gradients and screen-space arrow sampling),
`theme.js` (persisted theme and semantic map colors), `recording.js` (browser
capture and automatic MP4 upload), and `replay.js` (replay validation and time
lookup). Typography uses locally
installed Noto Sans SC / Microsoft YaHei and IBM Plex Mono / Consolas fallbacks;
there are no remote font requests that can stall an offline/local session.

The server API is `VisualServer(project_root, output_dir, host='127.0.0.1',
port=8787).start()`. `publish({'type':'scene', 'scene':scene, 'task':task,
'actor':actor})` begins an episode; `publish({'type':'frame', 'frame':frame})`
streams a frame. Bare frame dictionaries are also accepted. Call `close()` to
stop. The `url` and bound `port` properties are available; port 0 selects a free
port. Browser connections and queued frames are bounded. Slow viewers skip old
frames rather than blocking simulation.

`/events` is SSE. `/api/replays` lists NPZ and legacy JSON replays beneath the configured output
directory, excluding caches, hidden directories and logs. Files are served
read-only from `/visual/`, `/figs/`, and `/outputs/`; the root serves the UI.
GET on a replay `.npz` returns its JSON representation without decompressing
training observations or persisting a second trajectory. The archive SHA is
returned in `X-Replay-SHA256` for recording provenance. Local NPZ imports use
the bounded same-origin `POST /api/replays/convert` endpoint; local JSON imports
remain in the browser. NPZ uses numeric arrays and UTF-8 metadata, never pickle.
Traversal and symlinks escaping these roots are rejected.

`/api/visual-config` supplies the current entry-point display parameters. Only
known `visual_*` fields are accepted. Precedence is current server values, then
`scene.config_snapshot`, then compatible defaults. Switching replays retains
server overrides and user layer choices. A missing endpoint on an older server
falls back to the recorded configuration. The entry-point `full` trail style is
normalized to the UI's complete-history mode.

Every visual and metric is driven by actual replay frames or live events. World
coordinates use metres with +Y upward; headings are radians, CCW from +X. Model
images point along +X and rotate by `-atan2(vy,vx)` on the canvas. Coverage age
grid row 0 is south; negative cells are masked; `coverage.seen` distinguishes
unobserved cells. No example data or invented metrics are shipped.

The separate **UAV visited-cell layer** covers `world_bounds` with a default
250 m sampling grid. A sparse map stores last-visit simulation times only for
crossed cells. Segment traversal fills cells between streamed observations.
Visited cells fade over 600 simulated seconds from alpha .55 to .025; pausing
freezes that age. Rewinding rebuilds visits up to the selected frame, preventing
future traces. Display grid lines merge in whole-cell multiples when zoomed out;
the visited footprints retain the underlying physical cell size. This layer is
explicitly distinct from radar scan age and is never counted as radar coverage.

Trails can be hidden or shown as a default 60 s meteor tail or complete history
up to the selected frame. Meteor tails are clipped/interpolated exactly at the
time-window boundary. Model length and breadth use metres times map scale, with
default visible minima of 12 px for the UAV and 8 px for the ship. Transparent
source margins are cropped before sizing. The radar radius always equals the
recorded physical range times map scale; icon size never changes it.

Coverage opens at the task ROI; tracking follows the vessel. Full world,
follow-UAV, zoom and pan remain available. Historical playback supports pause,
seek, restart and 1–120× speed. Local JSON/NPZ imports are
listed alongside the recorded replays for the current task and policy.

Asset provenance, licenses and transformations are in `../figs/sources.json`
and `../figs/ATTRIBUTION.md`. The photographs are downloaded real images. The
loaded SVG files are original schematic 2D models, not downloaded 3D CAD.

## Attraction and repulsion field controls

The **引斥场色块** and **场方向** switches independently control the scalar
heatmap and direction arrows. Both use the actual current `frame.potential`.
The fixed signed scale is coral for hard constraints (-2), mauve for repulsion
(-1) and green for attraction (+1); greater absolute values have greater opacity.
Arrows follow **positive preference gradients**, toward higher values. They use
physical grid spacing with central differences, or one-sided differences beside
masked/missing cells. Constant regions have no arrows. Masked zeroes never act
as measured neighbors. Sampling adapts to zoom to keep arrows readable.

The **引斥场** view follows the UAV and fits its local field. The field radius
comes from sample bounds (normally 6 km); serialized `potential.range=[-2,1]`
is the scalar value range, not a physical radius. The field legend states its
sample resolution and radius; the separate radar legend states its actual
physical range (normally 3 km). Switching tasks/replays retains layer choices.
Replays without field data show a notice and disable the field camera.

## Top menus, themes and recordings

All task, data, view, playback and recording controls are in the top header.
The **数据** menu selects live/history sources, actors, replays and local imports.
The **视图** menu contains camera choices, every visualization switch, trail
style, zoom/reset and background theme. The **播放** menu contains the timeline,
speed, play/pause and restart. Menus work with keyboard focus, Space and Escape;
only one opens at a time. Mobile menus fit the viewport and scroll vertically.

White is the default theme; **视图 → 深海蓝** restores the dark ocean background.
The choice is remembered in local storage. Both palettes distinguish teal UAVs,
ochre ships, green islands, slate clouds, visits, radar and signed field values.
The field legend stays outside the map. Desktop monitoring scrolls independently.

In **录制**, turn on **录制演示** to capture actual replay or live frames. History
recordings advance through the loaded replay across the configured recording
duration. Turn the switch off to save early, or wait for automatic completion.
The server converts the browser capture to H.264 MP4 and saves the video plus
provenance metadata in its configured `output_dir/logs/recordings`. The menu shows
capture settings, progress, destination and the saved download link. Capture
uses the current map theme and layer choices. The `window.maritimeRecording`
automation API remains available.

On the next algorithm launch, pass `--experiment-name NAME` to place outputs,
datasets, weights, logs and independent evaluations below `outputs/NAME`.
Recordings are in `logs/recordings`, runtime caches in `logs/cache`, and TensorBoard logs below
`tf-logs/NAME`. `run.sh` uses `formal` by default; `--run-name` remains the run
directory inside each task's TensorBoard directory. Omit the new argument in
direct `main.py` calls to retain legacy paths. No running process is restarted.

## Verification

Verification on 2026-11-28 used Playwright CLI with Chromium-based Edge because
Browser/IAB was unavailable. Actual coverage and tracking preflight replays
were checked at 1440×1000, 1440×900 and 390×844. Both field switches changed the
canvas independently, seeking to the end and back restored the exact initial
bitmap, play/pause worked, and task changes preserved preferences. Checked white
default, dark persistence, all top menus, keyboard behavior, missing-field replay
compatibility and mobile overflow, with no browser errors. An isolated test
server captured and saved a real 640×400, 10 fps, 1.5 s H.264 MP4 through the
recording switch, and the returned video URL served successfully. Existing
training and TensorBoard processes were left running.
`view_image` inspection compared the original and updated screenshots for
layout, typography, palette, control states, map/legend placement and mobile
wrapping. No Image Gen tool was available; the existing UI/assets and the
written design specification were used as references.

Run the relevant regression suite with:

```sh
python -m pytest tests/test_main.py tests/test_field_visual.py tests/test_visual_frontend.py tests/test_visual_theme.py tests/test_visual_server.py tests/test_recording_ui.py tests/test_recording.py -q
```

These checks cover physical gradients, masks and thin boundaries, sparse visits,
replay rewind, physical model/radar sizing, theme persistence, recorder lifecycle,
server routes/transcoding, safe experiment names and output/log/cache isolation.
The JavaScript module tests use Node when available.
