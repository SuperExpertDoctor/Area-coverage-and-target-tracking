"""Run from the workspace root: python -m examples.plot_kinematics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamics import FixedWingModel, FixedWingState, ShipModel, ShipState


def run(output: Path) -> None:
    uav, ship = FixedWingModel(), ShipModel()
    u, s = FixedWingState(0, 0, 0), ShipState(0, 0, 0)
    dt = 0.1
    u_records, s_records = [], []
    for i in range(2401):
        t = i * dt
        heading = 0.0 if t < 10 else (np.pi / 2 if t < 100 else np.pi)
        vx, vy = uav.velocity(u)
        u_records.append((t, u.x, u.y, u.heading, u.bank, uav.yaw_rate(u), heading, np.hypot(vx, vy)))
        if i < 2400:
            u = uav.step_velocity(u, (np.cos(heading), np.sin(heading)), dt)
    for i in range(9001):
        t = i * dt
        heading = 0.0 if t < 30 else (np.pi / 2 if t < 420 else np.pi)
        vx, vy = ship.velocity(s)
        s_records.append((t, s.x, s.y, s.heading, s.rudder, s.yaw_rate, heading, np.hypot(vx, vy)))
        if i < 9000:
            s = ship.step_heading(s, heading, dt)
    u_data, s_data = np.asarray(u_records), np.asarray(s_records)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    fig.suptitle("Planar motion models: constant speed, finite steering response", fontsize=15)
    for col, data, title, color in [(0, u_data, "Fixed-wing UAV", "#2166ac"), (1, s_data, "Research vessel", "#1b7837")]:
        xy, heading_ax, actuator_ax = axes[:, col]
        xy.plot(data[:, 1] / 1000, data[:, 2] / 1000, color=color, lw=2)
        xy.scatter(data[0, 1] / 1000, data[0, 2] / 1000, marker="o", color=color, label="Start")
        xy.scatter(data[-1, 1] / 1000, data[-1, 2] / 1000, marker="x", color="#c44e52", label="End")
        xy.set(title=title, xlabel="East [km]", ylabel="North [km]", aspect="equal")
        xy.legend(loc="best", frameon=False)
        heading_ax.step(data[:, 0], np.rad2deg(data[:, 6]), where="post", color="#d98c10", ls="--", label="Heading setpoint")
        heading_ax.plot(data[:, 0], np.rad2deg(np.unwrap(data[:, 3])), color=color, label="Actual heading")
        heading_ax.set(xlabel="Time [s]", ylabel="Heading [deg]")
        heading_ax.legend(loc="best", frameon=False)
        actuator_ax.plot(data[:, 0], np.rad2deg(data[:, 4]), color=color)
        limit = uav.config.bank_limit if col == 0 else ship.config.max_rudder
        for sign in [-1, 1]:
            actuator_ax.axhline(sign * np.rad2deg(limit), color="#b44c4c", ls=":", lw=1)
        actuator_ax.set(xlabel="Time [s]", ylabel="Bank [deg]" if col == 0 else "Rudder [deg]")
        for ax in axes[:, col]:
            ax.grid(alpha=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    for name, data in [("uav", u_data), ("ship", s_data)]:
        np.savetxt(
            output.with_name(f"{output.stem}_{name}.csv"), data, delimiter=",",
            header="time_s,x_m,y_m,heading_rad,actuator_rad,yaw_rate_rad_s,heading_setpoint_rad,speed_m_s", comments="",
        )
    summary = {
        "assumptions": "2D, zero wind/current/sideslip, constant speed; maneuvering defaults are uncalibrated simulation parameters",
        "uav_config": asdict(uav.config),
        "ship_config": asdict(ship.config),
        "uav": {
            "speed_min_m_s": float(u_data[:, 7].min()), "speed_max_m_s": float(u_data[:, 7].max()),
            "max_bank_deg": float(np.rad2deg(np.abs(u_data[:, 4])).max()),
            "max_yaw_rate_deg_s": float(np.rad2deg(np.abs(u_data[:, 5])).max()),
            "min_instantaneous_radius_m": float(uav.config.speed / np.abs(u_data[:, 5]).max()),
        },
        "ship": {
            "speed_min_m_s": float(s_data[:, 7].min()), "speed_max_m_s": float(s_data[:, 7].max()),
            "max_rudder_deg": float(np.rad2deg(np.abs(s_data[:, 4])).max()),
            "max_yaw_rate_deg_s": float(np.rad2deg(np.abs(s_data[:, 5])).max()),
            "min_instantaneous_radius_m": float(ship.config.speed / np.abs(s_data[:, 5]).max()),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output.resolve())
    print(json.dumps({"uav": summary["uav"], "ship": summary["ship"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/kinematics/kinematics.png"))
    run(parser.parse_args().output)
