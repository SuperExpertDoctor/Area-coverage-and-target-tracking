const STORAGE_KEY = 'maritime-theme';
const themeName = value => value === 'dark' ? 'dark' : 'light';

export function readTheme(storage) {
  try {return themeName((storage ?? globalThis.localStorage)?.getItem(STORAGE_KEY));}
  catch (_) {return 'light';}
}

export function applyTheme(value, {document = globalThis.document, storage} = {}) {
  const name = themeName(value);
  document.documentElement.dataset.theme = name;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = name === 'dark' ? '#0b1420' : '#f2f5f9';
  try {(storage ?? globalThis.localStorage)?.setItem(STORAGE_KEY, name);} catch (_) { /* Private browsers can block storage. */ }
  return name;
}

const LIGHT = Object.freeze({
  overlay:'#ffffff00', grid:'#43627b19', gridText:'#58728a99', boundary:'#58728a70',
  roi:'#168d8080', roiText:'#16796e', visitGrid:'#42698a24',
  fieldArrow:'21,82,81', fieldOutline:'#ffffffe6', fieldGuide:'#3d817a80', goal:'#168663',
  cloudCenter:'#8d9da53b', cloudMid:'#a1b3c326', cloudEdge:'#b2ccda0b',
  cloudStroke:'#637b9199', cloudContour:'#6c859b70', cloudText:'#4d667e',
  islandFill:'#699b7930', islandStroke:'#54886ba6', islandSolid:'#91b19bd9', islandContour:'#5a856599', islandText:'#41694e',
  uav:'#087f7a', ship:'#ab6a1b', uavTrail:'8,127,122', shipTrail:'171,106,27', label:'#425c71',
  radarCenter:'#199d8e24', radarEdge:'#199d8e08', radarVisible:'#168d69cc', radarStroke:'#168a80aa', radarLine:'#168a8066', shadow:'#6c859b',
});
const DARK = Object.freeze({
  overlay:'#07172568', grid:'#819eb212', gridText:'#7794a867', boundary:'#7297ad55',
  roi:'#70ddd64a', roiText:'#9cc6cc', visitGrid:'#9cc5e021',
  fieldArrow:'208,247,239', fieldOutline:'#061724b3', fieldGuide:'#c1e7df66', goal:'#a2f5c5',
  cloudCenter:'#c4d8e044', cloudMid:'#a2bdcc20', cloudEdge:'#b2ccda08',
  cloudStroke:'#b3ccdb66', cloudContour:'#c2d9e340', cloudText:'#b6ccd8',
  islandFill:'#55786d2b', islandStroke:'#7da89b70', islandSolid:'#4b655cbb', islandContour:'#9aaf8f33', islandText:'#b3c2ae',
  uav:'#90eee3', ship:'#ffd197', uavTrail:'112,221,214', shipTrail:'247,188,117', label:'#a9c1d1',
  radarCenter:'#70ddd62b', radarEdge:'#70ddd60a', radarVisible:'#8df2b4aa', radarStroke:'#70ddd67a', radarLine:'#70ddd64a', shadow:'#03111e',
});

export const mapPalette = name => name === 'dark' ? DARK : LIGHT;
