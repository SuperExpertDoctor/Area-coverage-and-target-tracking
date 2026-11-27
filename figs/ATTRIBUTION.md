# Visualization asset credits

All downloaded images are real photographs. Full download URLs, source pages,
licenses, byte counts and SHA-256 checksums are recorded in `sources.json`.

* `ch4-uav.jpg`: **CH-4 at Airshow China Zhuhai 2022**, Infinty 0,
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:CH-4_at_Airshow_China_Zhuhai_2022.jpg),
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  Original file retained unchanged; cropped with CSS in the UI.
* `research-vessel.jpg`: **R/V Atlantis**, image courtesy of Kenneth Kostel,
  Woods Hole Oceanographic Institution,
  [USGS source explicitly labels this image Public Domain](https://www.usgs.gov/media/images/rv-atlantis).
  Original file retained unchanged; cropped with CSS in the UI.
* `ocean.jpg`: **Sea wave abstract texture**, Lukas Blazek / goumbik,
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sea_wave_abstract_texture_(Unsplash).jpg),
  [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
  Original retained unchanged. `ocean-map.jpg` is a resized, darkened and
  desaturated CC0 derivative for map readability. It is not a georeferenced chart.

`uav-model.svg` and `ship-model.svg` are hand-authored schematic **2D models**,
using the photographed platform families as visual references, with their forward
axis along +X. They are loaded as actual image assets by `visual/map.js`, then
rotated with `-atan2(vy, vx)` for the canvas Y-down coordinate system. When velocity
is zero, the recorded physical heading is used. These are not downloaded 3D CAD
models and are not engineering dimension drawings. Display sizes are enlarged
for legibility; map positions, obstacle radii and radar geometry use real metres.

The original schematic SVGs are available under CC0 1.0. The CH-4 photo remains
under its original CC BY-SA 4.0 license. No image-generation service was used.
