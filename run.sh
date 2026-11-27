#!/usr/bin/env bash
set -euo pipefail
# All tunable simulation, expert, training and export values live in this entry.
# Usage: bash run.sh [formal|all|expert|expert-validate|collect|train|dagger|evaluate|serve] [extra main.py flags]
# SIM_CONFIG.potential_enabled selects the protocol; --potential-field / --no-potential-field explicitly override it.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-python}"
MODE="${1:-formal}"
if (($#)); then shift; fi
OUTPUT_DIR="outputs/formal"
TF_LOG_DIR="tf-logs/formal"
RUN_NAME="formal-1000"
# Optional explicit inputs. Empty strings use the task's standard output paths.
CHECKPOINT_PATH=""
SPLIT_REFERENCE_CHECKPOINT=""
RESUME_CHECKPOINT=""
DATASET_MANIFEST=""
export PYTHONPYCACHEPREFIX="outputs/cache/python"
export MPLCONFIGDIR="outputs/cache/matplotlib"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
# Set before NumPy loads in the parent or spawned workers; training sets its
# separate explicit PyTorch --num-threads budget below.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
SIM_CONFIG=$(cat <<'JSON'
{
  "world_bounds": [
    0.0,
    0.0,
    300000.0,
    300000.0
  ],
  "roi_min_size": 10000.0,
  "roi_max_size": 20000.0,
  "roi_outer_buffer": 6000.0,
  "coverage_grid": 64,
  "potential_enabled": true,
  "potential_radius": 6000.0,
  "potential_period": 3600.0,
  "potential_decay": 3.0,
  "coverage_pan_limit": 1.0471975511965976,
  "potential_goal_hold_seconds": 180.0,
  "potential_goal_radius": 1500.0,
  "potential_unseen_bonus": 2.0,
  "field_heading_offsets": [0.0, -0.18, 0.18, -0.36, 0.36],
  "field_prediction_seconds": 8.0,
  "field_path_samples": 4,
  "field_guidance_weight": 2.0,
  "field_scan_weight": 1.4,
  "field_preference_weight": 0.2,
  "field_roi_weight": 0.3,
  "field_turn_weight": 0.05,
  "field_obstacle_weight": 1.0,
  "field_obstacle_influence": 2500.0,
  "field_coverage_obstacle_buffer": 200.0,
  "field_tracking_repulsion_limit": 0.12,
  "field_pan_samples": 9,
  "field_pan_goal_weight": 0.6,
  "field_pan_slew_weight": 0.05,
  "cloud_min_count": 2,
  "cloud_max_count": 6,
  "max_clouds": 8,
  "cloud_min_radius": 500.0,
  "cloud_max_radius": 1000.0,
  "island_min_count": 1,
  "island_max_count": 4,
  "max_islands": 8,
  "island_min_radius": 500.0,
  "island_max_radius": 2000.0,
  "min_water_fraction": 0.7,
  "uav_cloud_margin": 200.0,
  "ship_island_margin": 100.0,
  "navigation_grid_resolution": 250.0,
  "decision_dt": 1.0,
  "physics_dt": 0.1,
  "coverage_episode_seconds": 10800.0,
  "tracking_episode_seconds": 1800.0,
  "max_scene_attempts": 200,
  "per_scene_start_attempts": 100,
  "uav_speed": 61.11111111111111,
  "uav_min_turn_radius": 1000.0,
  "uav_max_bank": 0.4363323129985824,
  "uav_roll_time_constant": 1.0,
  "uav_max_roll_rate": 0.2617993877991494,
  "uav_heading_time_constant": 1.0,
  "uav_integration_step": 0.05,
  "ship_speed": 6.173333333333333,
  "ship_min_turn_radius": 300.0,
  "ship_nomoto_time_constant": 20.0,
  "ship_max_rudder": 0.6108652381980153,
  "ship_max_rudder_rate": 0.05235987755982989,
  "ship_rudder_time_constant": 1.0,
  "ship_heading_time_constant": 30.0,
  "ship_yaw_damping": 0.8,
  "ship_integration_step": 0.05,
  "coverage_range": 3000.0,
  "coverage_half_angle": 0.5235987755982988,
  "coverage_offset": -1.5707963267948966,
  "tracking_range": 3000.0,
  "tracking_half_angle": 0.5235987755982988,
  "pan_limit": 2.0943951023931953,
  "max_pan_rate": 0.5235987755982988,
  "tracking_desired_distance": 2000.0,
  "tracking_initial_min_distance": 1600.0,
  "tracking_initial_max_distance": 2400.0,
  "tracking_min_distance": 1300.0,
  "tracking_range_guard": 200.0,
  "tracking_angle_guard": 0.08726646259971647,
  "age_scale": 3600.0,
  "ship_lookahead": 700.0,
  "ship_route_spacing": 25.0,
  "ship_spiral_turns": 4.0,
  "ship_spiral_inner_radius": 30000.0,
  "ship_spiral_outer_radius": 125000.0,
  "coverage_lane_spacing": 2200.0,
  "coverage_sweep_timeout": 5400.0,
  "expert_lookahead": 650.0,
  "expert_obstacle_buffer": 1300.0,
  "tracking_radial_gain": 3.0,
  "coverage_outer_offset": 700.0,
  "coverage_corner_radius_factor": 1.35,
  "coverage_path_spacing": 100.0,
  "coverage_center_radius_factor": 1.4,
  "coverage_center_path_points": 140,
  "coverage_min_rectangle_factor": 2.8,
  "coverage_repair_timeout": 360.0,
  "coverage_repair_min_seconds": 160.0,
  "coverage_repair_capture_distance": 2500.0,
  "coverage_repair_hit_radius": 1500.0,
  "coverage_repair_orbit_radius": 1700.0,
  "coverage_repair_radial_gain": 2.5,
  "coverage_unseen_weight": 20.0,
  "coverage_age_weight": 1.0,
  "coverage_distance_scale": 15000.0,
  "coverage_recent_goal_factor": 0.1,
  "coverage_recent_goal_count": 4,
  "coverage_completion_index_margin": 12,
  "coverage_route_local_search_points": 65,
  "coverage_planning_blocks": 8,
  "coverage_rescan_gap_seconds": 60.0,
  "render_grid_stride": 4,
  "expert_safety_horizon": 30.0,
  "expert_safety_projection_dt": 5.0,
  "expert_safety_activation_distance": 2200.0,
  "expert_safety_enter_extension": 1700.0,
  "expert_safety_exit_extension": 2200.0,
  "expert_safety_radial_gain": 2.5,
  "expert_safety_inward_gain": 0.2,
  "expert_safety_forward_threshold": 0.45,
  "expert_world_buffer_turn_factor": 2.5,
  "expert_world_inward_gain": 3.0,
  "expert_safety_heading_offsets": [
    0.0,
    -0.4,
    0.4,
    -0.8,
    0.8,
    -1.4,
    1.4
  ],
  "expert_safety_clearance_reward": 5e-05,
  "expert_safety_clearance_cap": 1000.0,
  "expert_safety_extra_margin": 10.0,
  "ship_planning_radius_factor": 2.0,
  "ship_route_extra_margin": 500.0,
  "ship_astar_padding": 2000.0,
  "ship_bypass_smoothing_radius": 600.0,
  "ship_bypass_corner_fraction": 0.4,
  "ship_route_local_search_points": 80,
  "ship_spiral_generation_points": 6000,
  "ship_spiral_world_radius_factor": 0.42,
  "ship_spiral_inner_ratio": 0.4,
  "ship_bypass_lead_distance": 1000.0,
  "coverage_cloud_padding": 2500.0,
  "tracking_island_sampling_extent": 14000.0,
  "tracking_cloud_sampling_extent": 18000.0,
  "tracking_cloud_corridor_clearance": 3500.0,
  "tracking_corridor_sample_stride": 20,
  "tracking_corridor_lookahead_distance": 2500.0,
  "visual_grid_cell_size": 250.0,
  "visual_grid_fade_seconds": 600.0,
  "visual_grid_min_alpha": 0.025,
  "visual_grid_max_alpha": 0.55,
  "visual_grid_min_spacing_px": 12.0,
  "visual_uav_length": 9.0,
  "visual_uav_wingspan": 18.0,
  "visual_ship_length": 83.0,
  "visual_ship_beam": 16.0,
  "visual_uav_min_pixels": 12.0,
  "visual_ship_min_pixels": 8.0,
  "visual_trail_seconds": 60.0,
  "visual_show_grid": true,
  "visual_show_trail": true,
  "visual_show_radar_age": true,
  "visual_show_potential": true,
  "visual_trail_style": "meteor"
}
JSON
)
INPUT_ARGS=()
if [[ -n "$CHECKPOINT_PATH" ]]; then INPUT_ARGS+=(--checkpoint-path "$CHECKPOINT_PATH"); fi
if [[ -n "$SPLIT_REFERENCE_CHECKPOINT" ]]; then INPUT_ARGS+=(--split-reference-checkpoint "$SPLIT_REFERENCE_CHECKPOINT"); fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then INPUT_ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT"); fi
if [[ -n "$DATASET_MANIFEST" ]]; then INPUT_ARGS+=(--dataset-manifest "$DATASET_MANIFEST"); fi
exec "$PYTHON" main.py \
  --mode "$MODE" --task both \
  --config-json "$SIM_CONFIG" --field-mask-previous-command --field-frame body --field-cloud-geometry-encoder auto --field-pan-residual --tracking-pan-reference goal --coverage-pan-reference goal --coverage-direction-reference goal_orbit --tracking-direction-reference goal_orbit \
  --closed-loop-selection --selection-coverage-seconds 10800 --selection-tracking-seconds 1800 --selection-render-every 60 --selection-workers 2 --selection-tracking-pan-stress 0.3490658503988659 \
  --output-dir "$OUTPUT_DIR" --tf-log-dir "$TF_LOG_DIR" --run-name "$RUN_NAME" \
  --visualize --open-browser --keep-server --host 127.0.0.1 --port 8787 \
  --render-every 5 --realtime-factor 0 --progress-every 300 \
  --expert-seed 7100 --collection-seed 12000 \
  --episodes 1000 --collection-attempt-multiplier 3 \
  --collection-workers 8 --collection-replay-every 60 --collection-resume --collection-require-task-success \
  --collection-coverage-seconds 10800 --collection-tracking-seconds 1800 \
  --epochs 20 --batch-size 16 --sequence-length 128 \
  --learning-rate 0.0003 --weight-decay 0.00001 --grad-clip 1 \
  --validation-fraction 0.2 --training-seed 42 \
  --training-progress-every 50 --training-progress-seconds 30 --training-load-workers 4 \
  --hidden-size 128 --heading-residual --action-frame body --feature-frame auto --normalize-vectors --normalization-min-std 0.1 \
  --correction-weight-max 1 --correction-threshold 0.1 --field-steering-loss-weight 0 --dagger-recovery-min-steps 0 \
  --policy-safety-layer none --no-warm-start --no-coverage-ego-map --coverage-ego-map-sigma 2 --coverage-ego-map-mode gaussian \
  --no-coverage-mask-previous-command \
  --device auto --inference-device cpu --num-threads 4 --inference-num-threads 1 \
  --formal-test-episodes 8 --formal-test-seed 960000 --formal-test-workers 4 --formal-test-replay-every 60 \
  --formal-coverage-safety-layer forecast --formal-monitor-seconds 30 --formal-resume \
  --browser-recording --recording-headless --recording-fps 24 --recording-width 1440 --recording-height 900 \
  --recording-duration-seconds 30 --recording-bitrate 8000000 --recording-max-bytes 268435456 --recording-max-seconds 600 \
  --dagger-episodes 4 --dagger-iteration 1 --dagger-beta 0.8 --dagger-seed 24000 --dagger-initial-pan-jitter 0 --dagger-max-steps 0 --no-dagger-collect-only \
  --validation-coverage-seeds 2000 3000 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3012 3013 3014 3015 3016 3017 3018 3019 3020 3021 3022 3023 3024 3025 3026 3027 3028 \
  --validation-boundary-seeds 4000 4001 4002 --validation-boundary-starts 8 \
  --validation-tracking-seeds 5000 5001 5002 5003 5004 5005 5006 5007 5008 5009 \
  --validation-workers 4 --validation-coverage-seconds 10800 --validation-tracking-seconds 1800 \
  --validation-trajectory-every 60 --validation-coverage-ratio 1 --validation-resume \
  --save-video --video-fps 24 --video-max-frames 480 \
  --video-width 1440 --video-height 900 \
  "${INPUT_ARGS[@]}" \
  "$@"
