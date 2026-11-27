# Planar Kinematics Implementation Plan

> **For agentic workers:** Execute inline in this workspace using the test-first steps below. The user has requested implementation of the previously discussed models. No repository or existing code is present; do not create a Git repository for this task.

**Goal:** Implement reusable physically constrained planar UAV and ship motion.

**Architecture:** Immutable state records and pure state transitions in dynamics.py. The fixed wing uses coordinated turns and bounded roll response; the ship uses bounded rudder response and a first-order Nomoto model. All motion uses SI units and a shared adaptive-substep RK4 integrator.

**Tech Stack:** Python 3.10+, pytest; NumPy/Matplotlib for visualization only.

**Spec:** docs/superpowers/specs/2026-11-27-kinematics-design.md

## Global Constraints

- Two-dimensional positions; fixed speed magnitudes; no wind/current.
- Angles in radians, positive counterclockwise; x east/y north.
- Defaults: UAV 220 km/h, ship 12 knots; other maneuvering parameters are simulation assumptions.
- Future environment and experts use identical transitions and actuator states.
- Scope is the standalone motion component, tests, and diagnostic visualization.

## Task 1: Motion component and physical regression tests

Files: create dynamics.py and tests/test_dynamics.py.

Interfaces: FixedWingModel.step_velocity(state, velocity, dt), step_bank(state, bank, dt), velocity(state), yaw_rate(state); ShipModel.step_heading(state, heading, dt), step_rudder(state, rudder, dt), velocity(state). Frozen configs and states are exported from dynamics.py.

- [x] Add tests before production code. Analytic straight motion at V=10 m/s for 5 s from (3,4,pi/2) must end at (3,54).
- [x] Verify failure: `python -m pytest tests/test_dynamics.py -q` must fail because the module is absent. Observed 28 failures at the explicit missing-module assertion.
- [x] Implement coordinated turns, roll response, yaw autopilot, Nomoto response, steering response, bounded-state validation and RK4 substeps exactly as specified.
- [x] Verify the analytic Nomoto case T=20 s at t=20 s: r=r_inf*(1-exp(-1)), heading=r_inf*20*exp(-1). Full suite: 28 passed.

## Task 2: Reproducible diagnostic plot and usage document

Files: create examples/plot_kinematics.py, docs/kinematics.md and requirements.txt.

- [x] Generate actual UAV and ship trajectories with time series of roll/yaw/rudder, including heading-command responses.
- [x] Run `python -m examples.plot_kinematics --output artifacts/kinematics.png`; inspect the generated image. Verified smooth turns, gradual heading response, and actuator limits.
- [x] Document equations, physical assumptions, unit conventions, command/actual-state distinction, source links and environment integration example.
- [x] Verify `python -m pytest -q` (28 passed) and `python -m compileall -q dynamics.py tests examples` (exit 0). PNG, CSVs and JSON generated successfully.
