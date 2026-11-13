"""Task-local additions for Genesis 1.4.3; the installed simulator is not modified.

PBD's legacy rigid coupler resolves the table normal but has no cloth/table
friction. TableClothContact adds a split Coulomb velocity projection after each
scene step, using a weight-supported normal load. It is a reduced contact model,
not an exact friction-cone solve. It deliberately does not move positions or
alter fixed grasp particles. Extra normal load may be supplied explicitly.

Mass updates add absorbed liquid to cloth particles. Rendering helpers only
display supplied physical state; they do not modify the physical state.
"""

from __future__ import annotations

import numpy as np
import quadrants as qd
import torch

import genesis as gs


@qd.data_oriented
class TableClothContact:
    """GPU mass transfer and table friction for one built, unbatched PBD cloth.

    Call ``apply_friction(dt)`` immediately after ``scene.step()``. Coefficients
    and contact tolerance are explicit reconstruction assumptions. The table
    normal collision itself remains in Genesis. No historical scene data is
    used here.
    """

    def __init__(self, cloth, table_z=0.75, table_xy=(-0.8, 0.8, -0.5, 0.5),
                 mu_static=0.45, mu_kinetic=0.30, contact_tolerance=0.001):
        self.cloth = cloth
        self.solver = cloth.solver
        self.start = int(cloth.particle_start)
        self.count = int(cloth.n_particles)
        self.table_z = float(table_z)
        self.x_min, self.x_max, self.y_min, self.y_max = map(float, table_xy)
        self.radius = float(self.solver.particle_size) / 2.0
        self.mu_static = float(mu_static)
        self.mu_kinetic = float(mu_kinetic)
        self.tolerance = float(contact_tolerance)
        if self.solver._B != 1:
            raise ValueError("This reconstruction helper expects one environment.")
        masses = self.solver.particles_info.mass.to_numpy()[self.start:self.start + self.count]
        self.dry_masses = np.asarray(masses, dtype=np.float32).copy()
        self.current_masses = self.dry_masses.copy()
        self._edges = torch.as_tensor(cloth.edges, device=gs.device, dtype=torch.long)
        self._rest_lengths = torch.as_tensor(cloth._edges_len_rest, device=gs.device, dtype=gs.tc_float)
        self._diagnostics = qd.field(dtype=gs.qd_float, shape=5)
        self._free = qd.field(dtype=gs.qd_bool, shape=self.count)

    @qd.kernel
    def _set_masses(self, masses: qd.types.ndarray(), incoming_velocity: qd.types.ndarray()):
        for local in range(self.count):
            i = self.start + local
            previous_mass = self.solver.particles_info[i].mass
            added_mass = qd.max(0.0, masses[local] - previous_mass)
            if self.solver.particles[i, 0].free and added_mass > 0.0:
                incoming = qd.Vector([incoming_velocity[local, 0], incoming_velocity[local, 1], incoming_velocity[local, 2]])
                self.solver.particles[i, 0].vel = (
                    previous_mass * self.solver.particles[i, 0].vel + added_mass * incoming
                ) / masses[local]
            self.solver.particles_info[i].mass = masses[local]

    @qd.kernel
    def _redistribute_mass_momentum(self, masses: qd.types.ndarray(), delta_p: qd.types.ndarray()):
        for local in range(self.count):
            i = self.start + local
            old_mass = self.solver.particles_info[i].mass
            if self.solver.particles[i, 0].free:
                momentum = qd.Vector([delta_p[local, 0], delta_p[local, 1], delta_p[local, 2]])
                self.solver.particles[i, 0].vel = (
                    old_mass * self.solver.particles[i, 0].vel + momentum
                ) / masses[local]
            self.solver.particles_info[i].mass = masses[local]

    def set_absorbed_masses(self, absorbed_kg, incoming_velocity=None, momentum_delta=None):
        """Set cumulative absorbed kg per cloth particle, preserving dry mass.

        Source liquid removal must be performed by the film solver in exactly
        the same amount. Added mass mixes momentum in free cloth particles;
        incoming liquid velocity defaults to zero (stationary film). Fixed
        grasp particles keep their prescribed velocity, with the grasp acting
        as the momentum reservoir. With ``momentum_delta`` the caller supplies
        signed per-node momentum from conservative internal wicking; local wet
        masses may decrease, but total cloth liquid may not decrease. The update
        then uses (old momentum + supplied momentum change) / new mass.
        """
        if isinstance(absorbed_kg, torch.Tensor):
            absorbed_kg = absorbed_kg.detach().cpu().numpy()
        absorbed = np.asarray(absorbed_kg, dtype=np.float32)
        if absorbed.shape != (self.count,) or not np.isfinite(absorbed).all() or np.any(absorbed < 0):
            raise ValueError("absorbed_kg must contain one finite, nonnegative mass per cloth particle.")
        masses = np.ascontiguousarray(self.dry_masses + absorbed)
        if momentum_delta is not None:
            if incoming_velocity is not None:
                raise ValueError('Supply momentum_delta or incoming_velocity, not both')
            if masses.sum(dtype=np.float64) < self.current_masses.sum(dtype=np.float64)-1e-9:
                raise ValueError('Wicking cannot remove total cloth liquid mass')
            dp = _as_numpy(momentum_delta)
            if dp.shape != (self.count,3) or not np.isfinite(dp).all():
                raise ValueError('momentum_delta must be finite Nx3 kg*m/s')
            self.current_masses=masses
            self._redistribute_mass_momentum(masses,dp)
            return
        if np.any(masses < self.current_masses - 1e-9):
            raise ValueError("Local mass removal requires conservative momentum_delta.")
        if incoming_velocity is None:
            incoming_velocity = np.zeros((self.count, 3), dtype=np.float32)
        else:
            incoming_velocity = _as_numpy(incoming_velocity)
            incoming_velocity = np.ascontiguousarray(np.broadcast_to(incoming_velocity, (self.count, 3)))
            if not np.isfinite(incoming_velocity).all():
                raise ValueError("incoming_velocity must be finite.")
        self.current_masses = masses
        self._set_masses(self.current_masses, incoming_velocity)

    @qd.kernel
    def _friction(self, dt: qd.f32, extra_normal_acc: qd.f32):
        for k in range(5):
            self._diagnostics[k] = 0.0
        for local in range(self.count):
            i = self.start + local
            p = self.solver.particles[i, 0].pos
            v = self.solver.particles[i, 0].vel
            free = self.solver.particles[i, 0].free
            self._free[local] = free
            inside = (p[0] >= self.x_min and p[0] <= self.x_max and
                      p[1] >= self.y_min and p[1] <= self.y_max)
            # Receding particles more than 2 cm/s vertically are no longer
            # weight-supported contact. This avoids damping an upward lift.
            contact = (inside and p[2] <= self.table_z + self.radius + self.tolerance
                       and p[2] >= self.table_z - self.radius and v[2] <= 0.02)
            if contact:
                self._diagnostics[0] += 1.0
            if contact and free:
                speed = qd.sqrt(v[0] * v[0] + v[1] * v[1])
                normal_acc = 9.81 + extra_normal_acc
                dv = qd.min(speed, self.mu_kinetic * normal_acc * dt)
                if speed <= self.mu_static * normal_acc * dt:
                    dv = speed
                if speed > 1.0e-10:
                    ratio = qd.max(0.0, 1.0 - dv / speed)
                    self.solver.particles[i, 0].vel[0] = v[0] * ratio
                    self.solver.particles[i, 0].vel[1] = v[1] * ratio
                    mass = self.solver.particles_info[i].mass
                    self._diagnostics[1] += 0.5 * mass * speed * speed * (1.0 - ratio * ratio)
                    self._diagnostics[2] += mass * v[0] * (ratio - 1.0)
                    self._diagnostics[3] += mass * v[1] * (ratio - 1.0)
                self._diagnostics[4] += 1.0

    def apply_friction(self, dt, extra_normal_acc=0.0):
        if dt <= 0 or extra_normal_acc < 0:
            raise ValueError("dt must be positive and extra_normal_acc nonnegative.")
        self._friction(float(dt), float(extra_normal_acc))

    def diagnostics(self):
        """Read actual simulated state; call sparingly because scalar reads sync GPU."""
        p = self.cloth.get_particles_pos()
        v = self.cloth.get_particles_vel()
        if p.ndim == 3:
            p, v = p[0], v[0]
        lengths = torch.linalg.vector_norm(p[self._edges[:, 0]] - p[self._edges[:, 1]], dim=1)
        ratios = lengths / self._rest_lengths.clamp_min(1e-12)
        masses = torch.as_tensor(self.current_masses, device=p.device, dtype=p.dtype)
        inside = ((p[:, 0] >= self.x_min) & (p[:, 0] <= self.x_max) &
                  (p[:, 1] >= self.y_min) & (p[:, 1] <= self.y_max))
        penetration = (self.table_z + self.radius - p[inside, 2]).clamp_min(0)
        d = self._diagnostics.to_numpy()
        return {
            "cloth_particles": self.count,
            "cloth_mass_kg": float(masses.sum().item()),
            "cloth_absorbed_mass_kg": float((self.current_masses - self.dry_masses).sum()),
            "cloth_com_m": ((p * masses[:, None]).sum(dim=0) / masses.sum()).detach().cpu().tolist(),
            "cloth_z_min_m": float(p[:, 2].min().item()),
            "cloth_z_max_m": float(p[:, 2].max().item()),
            "cloth_speed_max_m_s": float(torch.linalg.vector_norm(v, dim=1).max().item()),
            "cloth_edge_stretch_max": float(ratios.max().item()),
            "cloth_edge_stretch_p95": float(torch.quantile(ratios, 0.95).item()),
            "cloth_edge_compression_min": float(ratios.min().item()),
            "cloth_table_penetration_max_m": float(penetration.max().item()) if penetration.numel() else 0.0,
            "cloth_contact_particles": int(d[0]),
            "cloth_free_contact_particles": int(d[4]),
            "table_friction_dissipation_J_last_step": float(d[1]),
            "table_friction_impulse_xy_Ns_last_step": [float(d[2]), float(d[3])],
            "cloth_state_finite": bool(torch.isfinite(p).all().item() and torch.isfinite(v).all().item()),
        }


def _as_numpy(value, dtype=np.float32):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def update_node_colors(context, node, colors, original_vertex_order=True):
    """Update existing RGBA attributes in Genesis' streaming rasterizer.

    Colors must already be enabled when the primitive first reaches OpenGL.
    For a plain colored PBD cloth that is the normal path. Textured cloth can
    additionally call ``prepare_cloth_colors`` immediately after scene.build.
    """
    primitive = node.mesh.primitives[0]
    rgba = _as_numpy(colors)
    if rgba.shape == (4,):
        rgba = np.broadcast_to(rgba, (len(primitive.positions), 4)).copy()
    elif original_vertex_order and primitive.vertex_mapping is not None:
        rgba = rgba[primitive.vertex_mapping]
    if rgba.shape != (len(primitive.positions), 4):
        raise ValueError(f"RGBA shape {rgba.shape} does not match primitive vertices.")
    if primitive.color_0 is None:
        raise RuntimeError("Call prepare_cloth_colors before the first camera render to enable colors.")
    primitive.color_0 = rgba
    # Color shares the generic vertex buffer with tangents/UVs. Position and
    # normals are separate streaming buffers in this Genesis version.
    fields = []
    if not primitive.stream_positions:
        fields.append(primitive.positions)
        if primitive.normals is not None:
            fields.append(primitive.normals)
    for name in ("tangents", "texcoord_0", "texcoord_1", "color_0"):
        attr = getattr(primitive, name)
        if attr is not None:
            fields.append(attr)
    context.jit.update_buffer(node, "vertex", np.ascontiguousarray(np.hstack(fields), dtype=np.float32))


def prepare_cloth_colors(scene, cloth, env_index=0):
    """Return cloth node; enable color multipliers before its first rendering."""
    context = scene.visualizer.context
    node = context.static_nodes[(env_index, cloth.uid)]
    primitive = node.mesh.primitives[0]
    if primitive.color_0 is None:
        if primitive._vaid is not None:
            raise RuntimeError("Color attributes must be enabled before the first render.")
        primitive.color_0 = np.ones((len(primitive.positions), 4), dtype=np.float32)
        primitive._buf_flags = None
    return node


class FilmMeshVisual:
    """Persistent grid mesh rendered from the external thin-film solver state.

    This is a visual representation, not a separate collision mesh. Supply
    vertices in metres, indexed triangle faces, and RGBA from physical wetness.
    For dry cells, lower vertices beneath the opaque tabletop, or use alpha.
    """

    def __init__(self, scene, vertices, faces, colors, name="soy_film", roughness=0.28):
        from genesis.ext import pyrender
        import trimesh

        self.context = scene.visualizer.context
        tri = trimesh.Trimesh(vertices=_as_numpy(vertices), faces=_as_numpy(faces, np.int32), process=False)
        tri.visual.vertex_colors = np.clip(_as_numpy(colors) * 255.0, 0, 255).astype(np.uint8)
        mesh = pyrender.Mesh.from_trimesh(tri, name=name, smooth=True, double_sided=True)
        primitive = mesh.primitives[0]
        primitive.material.roughnessFactor = float(roughness)
        primitive.material.metallicFactor = 0.0
        primitive.material.alphaMode = "BLEND"
        self.context.add_external_node(mesh)
        self.node = self.context.external_nodes[name]

    def update(self, vertices, colors=None):
        vertices = _as_numpy(vertices)
        primitive = self.node.mesh.primitives[0]
        if vertices.shape != primitive.positions.shape:
            raise ValueError("Film grid topology and vertex count must stay fixed.")
        primitive.positions = vertices
        self.context.jit.update_buffer(self.node, "pos", vertices)
        normals = self.context.jit.update_normal(self.node, vertices)
        if normals is not None:
            self.context.jit.update_buffer(self.node, "normal", normals)
        if colors is not None:
            update_node_colors(self.context, self.node, colors)
