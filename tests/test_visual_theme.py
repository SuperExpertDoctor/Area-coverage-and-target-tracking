"""Theme choices and semantic map palettes run through the real ES module."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_themes_default_to_light_remember_choice_and_tolerate_blocked_storage():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    uri = (Path(__file__).resolve().parents[1] / 'visual' / 'theme.js').as_uri()
    script = f"""
      import assert from 'node:assert/strict';
      import {{readTheme, applyTheme, mapPalette}} from {uri!r};
      const values=new Map(),storage={{getItem:k=>values.get(k),setItem:(k,v)=>values.set(k,v)}};
      const document={{documentElement:{{dataset:{{}}}},querySelector:()=>null}};
      assert.equal(readTheme(storage),'light');
      assert.equal(applyTheme('dark',{{document,storage}}),'dark');
      assert.equal(document.documentElement.dataset.theme,'dark');assert.equal(readTheme(storage),'dark');
      applyTheme('unexpected',{{document,storage}});assert.equal(readTheme(storage),'light');
      const blocked={{getItem(){{throw Error('blocked')}},setItem(){{throw Error('blocked')}}}};
      assert.equal(readTheme(blocked),'light');applyTheme('dark',{{document,storage:blocked}});
      assert.equal(document.documentElement.dataset.theme,'dark');
      for(const name of ['light','dark']){{
        const p=mapPalette(name);assert.notEqual(p.uav,p.ship);assert.notEqual(p.cloudStroke,p.islandStroke);
        assert.notEqual(p.fieldArrow,p.overlay);
      }}
      assert.notEqual(mapPalette('light').fieldArrow,mapPalette('dark').fieldArrow);
    """
    result = subprocess.run([node, '--input-type=module', '-e', script],
                            capture_output=True, text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stderr
