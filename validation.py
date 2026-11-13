"""Sampled numerical checks for the independently reconstructed simulation.

These checks do not establish real-world fidelity or validate inferred material
parameters. They inspect actual sampled states, not commanded trajectories.
Panda position bounds come from the installed Genesis panda.xml. Speed caps,
acceleration caution levels, cloth tolerances and task success criteria are
explicit engineering assumptions. No historical task data are consulted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass
class Thresholds:
    derivative_start_s: float = 0.2
    joint_tolerance_rad: float = 0.002
    finger_tolerance_m: float = 0.0005
    velocity_relative_tolerance: float = 0.01
    actuator_force_relative_tolerance: float = 0.01
    acceleration_caution_rad_s2: float = 10.0
    finger_acceleration_caution_m_s2: float = 2.0
    cloth_max_edge_ratio_fail: float = 1.25
    cloth_max_edge_ratio_caution: float = 1.08
    cloth_min_triangle_area_ratio_fail: float = 0.03
    cloth_total_area_ratio_fail: float = 1.30
    cloth_penetration_fail_m: float = 0.003
    cloth_sampled_speed_fail_m_s: float = 2.0
    grasp_centroid_error_fail_m: float = 0.008
    mass_relative_error_fail: float = 1e-5
    mass_absolute_error_fail_kg: float = 1e-10
    negative_mass_tolerance_kg: float = 1e-12
    residual_fraction_success: float = 0.005
    final_clearance_success_m: float = 0.05


class SimulationValidator:
    """Accumulate checks with small bounded diagnostic storage.

    ``q``, ``dq`` and ``torque`` are actual arrays shaped (2, 9), in local DOF
    order. Here ``torque`` MUST be ``get_dofs_control_force()``; an optional
    ``internal_torque`` can separately record ``get_dofs_force()``. The latter
    includes dynamic effects and must not be compared to actuator force caps.

    For grasp measurements, supply one list of particle indices per gripper and
    ``grasp_points`` as their expected world-space centroids. Include the local
    grasp-to-TCP offset in those expected positions. This is a geometric check,
    not evidence of a force-feasible frictional grasp.

    Call ``finish`` after the final state. If the loop ended early, pass
    ``completed=False`` rather than letting a partial run claim task success.
    """

    def __init__(self, rest_cloth, triangles, table_z=0.75, particle_radius=0.002,
                 table_xy=(-0.8, 0.8, -0.5, 0.5), thresholds=None,
                 joint_lower=None, joint_upper=None, velocity_limits=None,
                 actuator_limits=None):
        self.thresholds = thresholds or Thresholds()
        self.rest = as_numpy(rest_cloth).reshape(-1, 3).copy()
        self.faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        if not len(self.faces) or not len(self.rest) or not np.isfinite(self.rest).all():
            raise ValueError("A finite nonempty cloth rest mesh is required")
        if self.faces.min() < 0 or self.faces.max() >= len(self.rest):
            raise ValueError("Faces must index the actual PBD particle mesh")
        self.edges = np.unique(np.sort(np.concatenate((self.faces[:, [0, 1]],
            self.faces[:, [1, 2]], self.faces[:, [2, 0]])), axis=1), axis=0)
        self.rest_lengths = self._edge_lengths(self.rest)
        self.rest_areas = self._triangle_areas(self.rest)
        if np.any(self.rest_lengths <= 1e-12) or np.any(self.rest_areas <= 1e-14):
            raise ValueError("Rest mesh contains a degenerate edge or triangle")
        self.table_z, self.radius = float(table_z), float(particle_radius)
        self.table_xy = tuple(map(float, table_xy))
        self.lower = as_numpy(joint_lower if joint_lower is not None else
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0, 0])
        self.upper = as_numpy(joint_upper if joint_upper is not None else
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, .04, .04])
        self.velocity_limits = as_numpy(velocity_limits if velocity_limits is not None else
            [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, .2, .2])
        self.actuator_limits = as_numpy(actuator_limits if actuator_limits is not None else
            [87, 87, 87, 87, 12, 12, 12, 20, 20])
        self.sample_count = 0
        self.start_time = None
        self.end_time = None
        self.previous_t = None
        self.previous_dq = None
        self.previous_q = None
        self.previous_q_time = None
        self.previous_cloth = None
        self.previous_cloth_time = None
        self.events = {}
        self.extrema = {}
        self.phase_samples = {}
        self.measurements = dict(joints=0, velocities=0, actuator_forces=0,
            accelerations=0, internal_forces=0, cloth=0, grasps=0, film=0)
        self.max_abs_velocity = np.zeros((2, 9))
        self.max_abs_acceleration = np.zeros((2, 9))
        self.max_abs_actuator_force = np.zeros((2, 9))
        self.max_abs_internal_force = np.zeros((2, 9))
        self.final_cloth = None
        self.final_film = None
        self.initial_film_mass = None

    def _edge_lengths(self, p):
        return np.linalg.norm(p[self.edges[:, 1]] - p[self.edges[:, 0]], axis=1)

    def _triangle_areas(self, p):
        a = p[self.faces]
        return .5 * np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1)

    def _extreme(self, name, value, t, phase, maximum=True):
        value = float(value)
        previous = self.extrema.get(name)
        if previous is None or (value > previous["value"] if maximum else value < previous["value"]):
            self.extrema[name] = dict(value=value, time_s=float(t), phase=str(phase))

    def _event(self, code, t, phase, value, threshold, severity="failure"):
        record = self.events.setdefault(code, dict(severity=severity, count=0,
            first_time_s=float(t), first_phase=str(phase), first_value=float(value),
            threshold=float(threshold)))
        record["count"] += 1
        record["last_time_s"] = float(t)
        record["last_phase"] = str(phase)

    def _finite(self, name, value, t, phase):
        finite = bool(np.isfinite(value).all())
        if not finite:
            self._event("nonfinite_" + name, t, phase, np.count_nonzero(~np.isfinite(value)), 0)
        return finite

    @staticmethod
    def _robot_array(value, name):
        a = as_numpy(value)
        if a.shape != (2, 9):
            raise ValueError(f"{name} shape must be (2,9), got {a.shape}")
        return a

    def update(self, t, phase, q, dq, torque, cloth_pos, grasp_points=None,
               grasp_indices=None, grasp_active=False, film_diag=None,
               ddq=None, internal_torque=None):
        t = float(t)
        if not np.isfinite(t) or (self.end_time is not None and t <= self.end_time):
            raise ValueError("Sample times must be finite and strictly increasing")
        self.sample_count += 1
        self.start_time = t if self.start_time is None else self.start_time
        self.end_time = t
        self.phase_samples[str(phase)] = self.phase_samples.get(str(phase), 0) + 1
        th = self.thresholds
        q, dq, torque = (self._robot_array(v, name) for v, name in
            ((q, "q"), (dq, "dq"), (torque, "torque")))
        q_ok, dq_ok, torque_ok = [self._finite(n, a, t, phase) for n, a in
            (("q", q), ("dq", dq), ("actuator_force", torque))]
        if q_ok:
            self.measurements["joints"] += 1
            violation = np.maximum(self.lower - q, q - self.upper)
            tolerances = np.array([th.joint_tolerance_rad] * 7 + [th.finger_tolerance_m] * 2)
            max_ratio = np.max(violation / tolerances)
            self._extreme("joint_limit_tolerance_ratio_max", max_ratio, t, phase)
            if max_ratio > 1:
                self._event("joint_limits", t, phase, max_ratio, 1)
            if self.previous_q is not None and self.previous_q_time >= th.derivative_start_s:
                sampled_speed = np.abs(q - self.previous_q) / (t - self.previous_q_time)
                ratio = float(np.max(sampled_speed / self.velocity_limits))
                self._extreme("joint_displacement_speed_limit_ratio_max_after_startup", ratio, t, phase)
                if ratio > 1 + th.velocity_relative_tolerance:
                    self._event("joint_sampled_displacement_speed", t, phase, ratio, 1 + th.velocity_relative_tolerance)
            self.previous_q, self.previous_q_time = q.copy(), t
        if dq_ok:
            self.measurements["velocities"] += 1
            self.max_abs_velocity = np.maximum(self.max_abs_velocity, np.abs(dq))
            if t >= th.derivative_start_s:
                ratio = float(np.max(np.abs(dq) / self.velocity_limits))
                self._extreme("velocity_limit_ratio_max_after_startup", ratio, t, phase)
                if ratio > 1 + th.velocity_relative_tolerance:
                    self._event("joint_velocity", t, phase, ratio, 1 + th.velocity_relative_tolerance)
            if ddq is not None:
                acc = self._robot_array(ddq, "ddq")
            elif self.previous_dq is not None:
                acc = (dq - self.previous_dq) / (t - self.previous_t)
            else:
                acc = None
            if acc is not None and self._finite("ddq", acc, t, phase) and t >= th.derivative_start_s:
                self.measurements["accelerations"] += 1
                self.max_abs_acceleration = np.maximum(self.max_abs_acceleration, np.abs(acc))
                limits = np.array([th.acceleration_caution_rad_s2] * 7 +
                    [th.finger_acceleration_caution_m_s2] * 2)
                ratio = float(np.max(np.abs(acc) / limits))
                self._extreme("acceleration_caution_ratio_max_after_startup", ratio, t, phase)
                if ratio > 1:
                    self._event("acceleration_caution", t, phase, ratio, 1, "caution")
            self.previous_dq, self.previous_t = dq.copy(), t
        if torque_ok:
            self.measurements["actuator_forces"] += 1
            self.max_abs_actuator_force = np.maximum(self.max_abs_actuator_force, np.abs(torque))
            ratio = float(np.max(np.abs(torque) / self.actuator_limits))
            self._extreme("actuator_force_limit_ratio_max", ratio, t, phase)
            if ratio > 1 + th.actuator_force_relative_tolerance:
                self._event("actuator_force_limits", t, phase, ratio, 1 + th.actuator_force_relative_tolerance)
        if internal_torque is not None:
            force = self._robot_array(internal_torque, "internal_torque")
            if self._finite("internal_force", force, t, phase):
                self.measurements["internal_forces"] += 1
                self.max_abs_internal_force = np.maximum(self.max_abs_internal_force, np.abs(force))
        p = as_numpy(cloth_pos)
        if p.shape != self.rest.shape:
            raise ValueError(f"Cloth particle shape changed: {p.shape}, expected {self.rest.shape}")
        if self._finite("cloth_positions", p, t, phase):
            self.measurements["cloth"] += 1
            self.final_cloth = p.copy()
            if self.previous_cloth is not None and self.previous_cloth_time >= th.derivative_start_s:
                speed = float(np.linalg.norm(p - self.previous_cloth, axis=1).max() / (t - self.previous_cloth_time))
                self._extreme("cloth_sampled_speed_max_m_s_after_startup", speed, t, phase)
                if speed > th.cloth_sampled_speed_fail_m_s:
                    self._event("cloth_sampled_speed", t, phase, speed, th.cloth_sampled_speed_fail_m_s)
            self.previous_cloth, self.previous_cloth_time = p.copy(), t
            edges = self._edge_lengths(p) / self.rest_lengths
            areas = self._triangle_areas(p) / self.rest_areas
            total_area_ratio = float(np.sum(areas * self.rest_areas) / np.sum(self.rest_areas))
            self._extreme("cloth_edge_ratio_max", edges.max(), t, phase)
            self._extreme("cloth_edge_ratio_p95_max", np.percentile(edges, 95), t, phase)
            self._extreme("cloth_edge_ratio_min", edges.min(), t, phase, maximum=False)
            self._extreme("cloth_triangle_area_ratio_min", areas.min(), t, phase, maximum=False)
            self._extreme("cloth_triangle_area_ratio_max", areas.max(), t, phase)
            self._extreme("cloth_total_area_ratio_max", total_area_ratio, t, phase)
            self._extreme("cloth_z_min_m", p[:, 2].min(), t, phase, maximum=False)
            if edges.max() > th.cloth_max_edge_ratio_fail:
                self._event("cloth_excessive_stretch", t, phase, edges.max(), th.cloth_max_edge_ratio_fail)
            elif edges.max() > th.cloth_max_edge_ratio_caution:
                self._event("cloth_stretch_caution", t, phase, edges.max(), th.cloth_max_edge_ratio_caution, "caution")
            if areas.min() < th.cloth_min_triangle_area_ratio_fail:
                self._event("cloth_degenerate_triangles", t, phase, areas.min(), th.cloth_min_triangle_area_ratio_fail)
            if total_area_ratio > th.cloth_total_area_ratio_fail:
                self._event("cloth_total_area_growth", t, phase, total_area_ratio, th.cloth_total_area_ratio_fail)
            xmin, xmax, ymin, ymax = self.table_xy
            on_table = (p[:, 0] >= xmin) & (p[:, 0] <= xmax) & (p[:, 1] >= ymin) & (p[:, 1] <= ymax)
            penetration = max(0., float(np.max(self.table_z + self.radius - p[on_table, 2]))) if on_table.any() else 0.
            self._extreme("cloth_table_penetration_max_m", penetration, t, phase)
            if penetration > th.cloth_penetration_fail_m:
                self._event("cloth_table_penetration", t, phase, penetration, th.cloth_penetration_fail_m)
            if grasp_active and grasp_points is not None and grasp_indices is not None:
                targets = as_numpy(grasp_points).reshape(-1, 3)
                if len(targets) != len(grasp_indices):
                    raise ValueError("One expected centroid per grasp index list is required")
                centroids = np.array([p[np.atleast_1d(ids).astype(int)].mean(0) for ids in grasp_indices])
                distances = np.linalg.norm(centroids - targets, axis=1)
                if self._finite("grasp_errors", distances, t, phase):
                    self.measurements["grasps"] += 1
                    self._extreme("grasp_centroid_error_max_m", distances.max(), t, phase)
                    if distances.max() > th.grasp_centroid_error_fail_m:
                        self._event("grasp_geometry", t, phase, distances.max(), th.grasp_centroid_error_fail_m)
        if film_diag is not None:
            self._update_film(film_diag, t, phase)

    def _update_film(self, d, t, phase):
        th = self.thresholds
        required = ["initial_liquid_mass_kg", "free_liquid_mass_kg", "absorbed_liquid_mass_kg"]
        if any(key not in d for key in required):
            raise ValueError("Film diagnostics require initial/free/absorbed_liquid_mass_kg")
        all_numeric = np.array([float(v) for v in d.values()
            if isinstance(v, (float, int, np.number))])
        if not self._finite("film_diagnostics", all_numeric, t, phase):
            return
        numbers = np.array([float(d[k]) for k in required])
        if not self._finite("film_mass", numbers, t, phase):
            return
        initial, free, absorbed = numbers
        escaped = float(d.get("escaped_liquid_mass_kg", 0.))
        if not np.isfinite(escaped):
            self._event("nonfinite_film_mass", t, phase, 1, 0)
            return
        self.measurements["film"] += 1
        self.final_film = {k: (float(v) if isinstance(v, (float, int, np.number)) else v) for k, v in d.items()}
        if initial <= 0:
            self._event("nonpositive_initial_film_mass", t, phase, initial, 0)
            return
        if self.initial_film_mass is None:
            self.initial_film_mass = initial
        elif abs(initial - self.initial_film_mass) > th.mass_absolute_error_fail_kg:
            self._event("initial_mass_changed", t, phase, abs(initial - self.initial_film_mass), th.mass_absolute_error_fail_kg)
        error = abs(free + absorbed + escaped - initial)
        rel = error / initial
        self._extreme("liquid_relative_mass_error_max", rel, t, phase)
        self._extreme("liquid_absolute_mass_error_max_kg", error, t, phase)
        if error > max(th.mass_absolute_error_fail_kg, th.mass_relative_error_fail * initial):
            self._event("liquid_mass_conservation", t, phase, rel, th.mass_relative_error_fail)
        if min(free, absorbed, escaped) < -th.negative_mass_tolerance_kg:
            self._event("negative_liquid_mass", t, phase, min(free, absorbed, escaped), -th.negative_mass_tolerance_kg)
        if "max_saturation" in d and float(d["max_saturation"]) > 1 + 1e-6:
            self._event("cloth_capacity_exceeded", t, phase, d["max_saturation"], 1 + 1e-6)
        if "min_height_m" in d and float(d["min_height_m"]) < -1e-12:
            self._event("negative_film_height", t, phase, d["min_height_m"], -1e-12)

    def finish(self, output_path=None, completed=True, metadata=None):
        failures = {k: v for k, v in self.events.items() if v["severity"] == "failure"}
        cautions = {k: v for k, v in self.events.items() if v["severity"] == "caution"}
        missing = [k for k in ("joints", "velocities", "actuator_forces", "cloth", "grasps", "film")
            if not self.measurements[k]]
        clearance = None if self.final_cloth is None else float(self.final_cloth[:, 2].min() - self.radius - self.table_z)
        valid_final_mass = self.final_film is not None and self.final_film["initial_liquid_mass_kg"] > 0
        residual = float(self.final_film["free_liquid_mass_kg"] / self.final_film["initial_liquid_mass_kg"]) if valid_final_mass else None
        absorbed_fraction = float(self.final_film["absorbed_liquid_mass_kg"] / self.final_film["initial_liquid_mass_kg"]) if valid_final_mass else None
        task_success = bool(completed and not failures and not missing and
            residual is not None and residual <= self.thresholds.residual_fraction_success and
            absorbed_fraction is not None and absorbed_fraction >= 1 - self.thresholds.residual_fraction_success and
            clearance is not None and clearance >= self.thresholds.final_clearance_success_m)
        result: dict[str, Any] = dict(
            schema="independent-reconstruction-sampled-validation-v1",
            completed=bool(completed), sample_count=self.sample_count,
            first_sample_time_s=self.start_time, last_sample_time_s=self.end_time,
            sampled_numerical_checks_passed=bool(self.sample_count and not failures and not missing),
            task_success_under_stated_criteria=task_success,
            phase_sample_counts=self.phase_samples, measurement_counts=self.measurements,
            missing_required_measurements=missing, failures=failures, cautions=cautions,
            extrema=self.extrema, thresholds=asdict(self.thresholds),
            final_cloth_clearance_m=clearance, final_free_liquid_fraction=residual,
            final_absorbed_liquid_fraction=absorbed_fraction,
            final_film_diagnostics=self.final_film,
            robot_limits=dict(joint_lower=self.lower.tolist(), joint_upper=self.upper.tolist(),
                velocity=self.velocity_limits.tolist(), actuator_force=self.actuator_limits.tolist()),
            actual_max_abs_joint_velocity=self.max_abs_velocity.tolist(),
            actual_max_abs_joint_acceleration_after_startup=self.max_abs_acceleration.tolist(),
            actual_max_abs_actuator_control_force=self.max_abs_actuator_force.tolist(),
            actual_max_abs_internal_generalized_force=self.max_abs_internal_force.tolist(),
            interpretation=[
                "These are sampled numerical consistency and task checks, not experimental validation.",
                "Joint position bounds are from installed Genesis panda.xml. Velocity and acceleration thresholds are declared assumptions; acceleration exceedance is a caution.",
                "Arm joint units are rad, rad/s, rad/s², Nm; finger units are m, m/s, m/s², N.",
                "Actuator limit checks use control force, not generalized internal force or reaction force.",
                "Cloth radius is a numerical collision shell, not the measured cotton thickness.",
                "Grasp error only checks attachment geometry; it does not establish finger friction or force feasibility.",
                "No self-intersection, fabric tear, material-identification, contact-force or complete continuous-time collision certification is provided.",
                "A passing cleaning criterion permits the reported residual mass; it does not imply exactly zero liquid.",
                "Startup derivative checks exclude the first declared time interval; its states and control-force caps are still checked.",
            ], metadata=metadata or {})
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return result


# Short alias for callers that prefer a generic utility name.
Validator = SimulationValidator
