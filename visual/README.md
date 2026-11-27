# Maritime visualizer

No frontend build or npm install is required. Start the project CLI with its
visual option, or run `python src/visual_server.py --port 8787`, then open
`http://127.0.0.1:8787/`.

The static frontend consists of `index.html`, `styles.css`, `app.js` (controls,
SSE and measured metrics), `map.js` (Cartesian map and loaded 2D model assets),
`layers.js` (sparse temporal visits, clipped trails and physical model sizing),
and `replay.js` (replay validation and time lookup). Typography uses locally
installed Noto Sans SC / Microsoft YaHei and IBM Plex Mono / Consolas fallbacks;
there are no remote font requests that can stall an offline/local session.

The server API is `VisualServer(project_root, output_dir, host='127.0.0.1',
port=8787).start()`. `publish({'type':'scene', 'scene':scene, 'task':task,
'actor':actor})` begins an episode; `publish({'type':'frame', 'frame':frame})`
streams a frame. Bare frame dictionaries are also accepted. Call `close()` to
stop. The `url` and bound `port` properties are available; port 0 selects a free
port. Browser connections and queued frames are bounded. Slow viewers skip old
frames rather than blocking simulation.

`/events` is SSE. `/api/replays` lists replay JSON beneath the configured output
directory, excluding caches, hidden directories and logs. Files are served
read-only from `/visual/`, `/figs/`, and `/outputs/`; the root serves the UI.
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
seek, restart and 1–120× speed. Local JSON imports stay in the browser and are
listed alongside the recorded replays for the current task and policy.

Asset provenance, licenses and transformations are in `../figs/sources.json`
and `../figs/ATTRIBUTION.md`. The photographs are downloaded real images. The
loaded SVG files are original schematic 2D models, not downloaded 3D CAD.

## Verification

Run `python -m pytest tests/test_visual_frontend.py -q` for 11 checks against the
actual ES modules: sparse segment traversal, world clipping, replay rewind,
live append/trim, revisits, simulation-time fade, minimum-size physical models,
actual model rotation/radar radius, meteor-window clipping, missing positions,
and server configuration surviving scene changes. These tests use Node when
available. Server route/SSE tests remain in `tests/test_visual_server.py`.

Playwright CLI Chromium was used because Browser/IAB tools were unavailable.
The latest grid/trail UI was inspected with `view_image` at 1440×900 and 390×844,
including full-world view and the mobile lower panel. Checked grid alignment,
separate visit/radar colors, Chinese/control typography, real-photo framing,
small velocity-oriented models, label spacing and responsive layout. Desktop
playback remains in view and mobile has no horizontal overflow. Browser caches,
snapshots and screenshots are kept beneath `outputs`; current review images are
in `outputs/diagnostics/visual-grid/`.

Verified the current real coverage replay at t=1300 s (53.0%), pause stability,
forward/backward seeking without residual future cells, meteor/history/hidden
trail states, and old replays adopting current display overrides across task
changes. The radar stays at 3 km. No Image Gen tool was available, so no
generated concept or comparison
against a generated concept is claimed.

The final browser recordings are `coverage-expert-grid.webm` (9.52 s) and
`tracking-policy-grid.webm` (10.00 s), both 1440×900, beneath
`outputs/diagnostics/visual-grid/`. These show the real JavaScript UI playing
recorded frames, including actual metrics, fading visited cells and meteor tails.
