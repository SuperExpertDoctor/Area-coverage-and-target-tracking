"""Entry-point completeness and meaningful configuration effects."""
from dataclasses import fields
import json
from pathlib import Path
import re

import numpy as np
import pytest

from config import Config


def test_run_sh_declares_every_simulation_and_visual_configuration_field():
    script=(Path(__file__).parents[1]/'run.sh').read_text(encoding='utf-8')
    config=json.JSONDecoder().raw_decode(script[script.index('\n{')+1:])[0]
    assert set(config)=={field.name for field in fields(Config)}
    assert Config.from_dict(config).to_dict()==config


def test_run_sh_exposes_every_cli_hyperparameter():
    import main
    script=(Path(__file__).parents[1]/'run.sh').read_text(encoding='utf-8')
    options=set(re.findall(r'--[a-zA-Z0-9-]+',script))
    for action in main.parser()._actions:
        if action.dest=='help':continue
        assert options.intersection(action.option_strings),action.dest


def test_visual_boolean_switches_and_style_are_validated():
    config=Config(visual_show_grid=False,visual_show_trail=False,visual_show_radar_age=False,
                  visual_trail_style='full')
    assert not config.visual_show_grid and not config.visual_show_trail
    with pytest.raises(ValueError,match='boolean'):Config(visual_show_grid=1)
    with pytest.raises(ValueError,match='trail_style'):Config(visual_trail_style='unknown')
    with pytest.raises(ValueError,match='alpha'):Config(visual_grid_min_alpha=.8,visual_grid_max_alpha=.5)


def test_tracking_angle_guard_changes_accepted_initial_envelope():
    from scenario import sample_tracking_scene,validate_scene,SceneValidationError
    config=Config()
    scene=sample_tracking_scene(config,np.random.default_rng(7),7)
    # Still inside the physical 30-degree half FOV; outside its guarded interior.
    scene['uav_initial']['heading']+=config.tracking_half_angle-config.tracking_angle_guard/2
    with pytest.raises(SceneValidationError,match='guarded radar'):validate_scene(scene,config,'tracking')
    validate_scene(scene,Config(tracking_angle_guard=0),'tracking')
    with pytest.raises(ValueError,match='guarded radar'):Config(tracking_range_guard=700)


def test_display_preferences_do_not_change_physical_observations_or_motion():
    from env_tracking import TrackingEnv
    left=TrackingEnv(Config())
    right=TrackingEnv(Config(visual_ship_length=30,visual_uav_min_pixels=5,
        visual_grid_cell_size=500,visual_show_grid=False,visual_show_trail=False))
    try:
        a,_=left.reset(seed=7);b,_=right.reset(seed=7)
        np.testing.assert_array_equal(a['vector'],b['vector'])
        for _ in range(3):
            action=np.array([np.cos(left.uav.heading),np.sin(left.uav.heading)],np.float32)
            a=left.step(action)[0];b=right.step(action)[0]
            np.testing.assert_array_equal(a['vector'],b['vector'])
            assert left.uav==right.uav and left.ship==right.ship
    finally:left.close();right.close()
