"""Conservative CUDA thin-film / absorbent-cloth reduced model.

This is a separately implemented reduced fluid model coupled to *observed* Genesis
PBD cloth positions, not Genesis' SPH/PBD liquid solver.  It cannot identify true
material properties from photographs.  All default material values are explicit
assumptions, and fluid forces are not returned to Genesis.  The small fluid load
can be returned as ``absorbed_mass`` for a mechanical coupling if desired.

Film equation (horizontal impermeable table):
    dh/dt + div(q) = -s,
    q = -h**3/(3*mu) grad(rho*g*h - gamma*laplacian(h)) + h*u,
where u is one half of the local contacting cloth's tangential velocity (Couette
approximation).  Upwind finite-volume face transfers have a donor-volume limiter;
closed domain boundaries and matching uptake/absorption transfers conserve mass.

Uptake at cloth material node i: rho*A_i*k*(1 - m_i/C_i), only while its collision
shell touches the local film, limited by local remaining liquid and capacity C_i.
Absorbed liquid also wicks along the cloth's material mesh, from larger to smaller
saturation.  Each pairwise exchange is limited by donor mass and receiver capacity
and transports the donor node's liquid momentum.  The effective in-plane wicking
diffusivity is an explicit phenomenological assumption, not a measured value.
No disappearance timer, evaporation, or cleanup threshold changes the physical
state.  A rendering threshold may hide subpixel film but diagnostics include it.

Thin-film basis: Kondic, SIAM Review 45, 95–115 (2003),
https://harp.njit.edu/~kondic/thin_films/SIAM_lub2003.pdf
The uptake law is a phenomenological, capacity-limited approximation, not a
calibrated cotton/soy-sauce constitutive law or a resolved pore-scale model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh


class ThinFilmAbsorption:
    def __init__(
        self,
        asset_path,
        cloth_rest_vertices,
        cloth_faces,
        table_z=0.75,
        puddle_center=(0.0, 0.04),
        puddle_scale=0.5,
        domain_center=(0.0, 0.0),
        domain_size=(0.62, 0.68),
        grid_shape=(80, 88),  # (nx, ny)
        device="cuda",
        density=1150.0,  # kg/m³; assumption, not recovered from images
        dynamic_viscosity=0.008,  # Pa s
        surface_tension=0.050,  # N/m
        cotton_areal_density=0.22,  # kg/m², 11.088 g for supplied 28x18cm mesh
        capacity_ratio=3.0,  # kg fluid / kg dry textile
        uptake_speed=0.00030,  # m/s equivalent liquid uptake speed, uncalibrated
        contact_radius=0.002,  # numerical PBD collision shell, NOT cloth thickness
        contact_tolerance=0.00025,
        shear_fraction=0.5,
        wicking_diffusivity=2e-5,  # m²/s effective cotton capillary spreading assumption
        dtype=torch.float64,
    ):
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for film model, but unavailable")
        self.device, self.dtype = torch.device(device), dtype
        self.rho, self.mu, self.gamma = density, dynamic_viscosity, surface_tension
        self.table_z = float(table_z)
        self.nx, self.ny = map(int, grid_shape)
        self.dx, self.dy = domain_size[0] / self.nx, domain_size[1] / self.ny
        self.cell_area = self.dx * self.dy
        self.xmin = domain_center[0] - domain_size[0] / 2
        self.ymin = domain_center[1] - domain_size[1] / 2
        x = self.xmin + (np.arange(self.nx) + 0.5) * self.dx
        y = self.ymin + (np.arange(self.ny) + 0.5) * self.dy
        yy, xx = np.meshgrid(y, x, indexing="ij")
        self.xy = torch.as_tensor(np.stack((xx, yy), axis=-1), device=self.device, dtype=dtype)
        vertices = np.asarray(cloth_rest_vertices, dtype=np.float64)
        faces = np.asarray(cloth_faces, dtype=np.int64)
        triangles = vertices[faces]
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
        node_area = np.zeros(len(vertices))
        np.add.at(node_area, faces.reshape(-1), np.repeat(area / 3, 3))
        self.node_area = torch.as_tensor(node_area, device=self.device, dtype=dtype)
        self.dry_node_mass = self.node_area * cotton_areal_density
        self.capacity = self.dry_node_mass * capacity_ratio
        self.absorbed_mass = torch.zeros_like(self.node_area)
        # Undirected material edges remain connected as the cloth folds/moves.
        edge_pairs = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edge_pairs = np.unique(np.sort(edge_pairs, axis=1), axis=0)
        length_squared = np.sum((vertices[edge_pairs[:, 0]]-vertices[edge_pairs[:, 1]])**2, axis=1)
        edge_pairs = edge_pairs[length_squared > 1e-20]
        length_squared = length_squared[length_squared > 1e-20]
        self.wick_edges = torch.as_tensor(edge_pairs, device=self.device, dtype=torch.long)
        self.wicking_diffusivity = float(wicking_diffusivity)
        if self.wicking_diffusivity < 0:
            raise ValueError("wicking_diffusivity must be nonnegative")
        cap_per_area = cotton_areal_density * capacity_ratio
        conductance = self.wicking_diffusivity * (node_area[edge_pairs[:, 0]]+node_area[edge_pairs[:, 1]]) / (2*length_squared) * cap_per_area
        self.wick_conductance = torch.as_tensor(conductance, device=self.device, dtype=dtype)
        node_conductance = torch.zeros_like(self.node_area)
        node_conductance.scatter_add_(0, self.wick_edges[:, 0], self.wick_conductance)
        node_conductance.scatter_add_(0, self.wick_edges[:, 1], self.wick_conductance)
        self.wick_max_rate = float((node_conductance/self.capacity.clamp_min(1e-20)).max().item())
        self.momentum_transfer = torch.zeros((len(vertices), 3), device=self.device, dtype=dtype)
        self.last_wicked_mass = 0.0
        self.last_wick_substeps = 0
        self.contact_radius, self.contact_tolerance = float(contact_radius), float(contact_tolerance)
        self.uptake_speed, self.shear_fraction = float(uptake_speed), float(shear_fraction)
        self.previous_positions = None
        self.elapsed = 0.0
        self.steps = 0
        self.last_contact_area = 0.0
        self.last_uptake = 0.0
        self.last_substeps = 0
        self.max_transport_limiter_fraction = 0.0
        self.transport_limited_substeps = 0
        self.height = torch.as_tensor(self._rasterize(asset_path, puddle_center, puddle_scale, xx, yy), device=self.device, dtype=dtype)
        self.initial_mass = float((self.height.sum() * self.cell_area * self.rho).item())
        if self.initial_mass <= 0:
            raise ValueError("Puddle does not intersect the film domain")
        ids = np.arange(self.nx * self.ny).reshape(self.ny, self.nx)
        self.film_faces = np.stack((
            np.stack((ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:]), -1),
            np.stack((ids[:-1, :-1], ids[1:, 1:], ids[1:, :-1]), -1)
        ), axis=2).reshape(-1, 3)
        self.parameters = dict(density=density, dynamic_viscosity=dynamic_viscosity,
            surface_tension=surface_tension, cotton_areal_density=cotton_areal_density,
            capacity_ratio=capacity_ratio, uptake_speed=uptake_speed,
            contact_radius=contact_radius, contact_tolerance=contact_tolerance,
            shear_fraction=shear_fraction, wicking_diffusivity=wicking_diffusivity,
            grid_shape=list(grid_shape),
            domain_size=list(domain_size), table_z=table_z,
            puddle_center=list(puddle_center), puddle_scale=puddle_scale,
            coupling="film receives cloth position/velocity; no fluid reaction force returned")

    @staticmethod
    def _rasterize(asset_path, center, scale, xx, yy):
        """Rasterize actual upward-facing asset triangles; retain its irregular outline."""
        mesh = trimesh.load(asset_path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(scale)
        vertices[:, :2] += np.asarray(center)
        height = np.zeros_like(xx)
        for face in np.asarray(mesh.faces)[np.asarray(mesh.face_normals)[:, 2] > 0.5]:
            a, b, c = vertices[face]
            den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
            if abs(den) < 1e-16:
                continue
            u = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / den
            v = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / den
            inside = (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)
            z = u*a[2] + v*b[2] + (1-u-v)*c[2]
            height[inside] = np.maximum(height[inside], z[inside])
        return height

    @property
    def wetness(self):
        return self.absorbed_mass / self.capacity.clamp_min(1e-20)

    def _links(self, pos):
        """Four nonnegative bilinear links from each cloth material node to cells."""
        gx = (pos[:, 0] - self.xmin) / self.dx - 0.5
        gy = (pos[:, 1] - self.ymin) / self.dy - 0.5
        ix, iy = torch.floor(gx).long(), torch.floor(gy).long()
        fx, fy = gx - ix, gy - iy
        cols = torch.stack((ix, ix+1, ix, ix+1), -1)
        rows = torch.stack((iy, iy, iy+1, iy+1), -1)
        valid = (cols >= 0) & (cols < self.nx) & (rows >= 0) & (rows < self.ny)
        weights = torch.stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy), -1) * valid
        indices = (rows.clamp(0, self.ny-1)*self.nx + cols.clamp(0, self.nx-1))
        bottom = pos[:, 2, None] - self.contact_radius
        film_surface = self.table_z + self.height.flatten()[indices]
        contact = (bottom <= film_surface + self.contact_tolerance) & (bottom >= self.table_z - 0.003)
        # The lower bound prevents uptake if invalid cloth tunnels through the table.
        return indices, weights * contact

    def _cloth_velocity(self, pos, dt, indices, weights, supplied_velocity=None):
        if supplied_velocity is None:
            velocity = torch.zeros_like(pos) if self.previous_positions is None else (pos - self.previous_positions) / dt
        else:
            velocity = torch.as_tensor(supplied_velocity, device=self.device, dtype=self.dtype)
            if velocity.shape != pos.shape:
                raise ValueError(f"Expected cloth velocities {tuple(pos.shape)}, got {tuple(velocity.shape)}")
        self.previous_positions = pos.clone()
        field = torch.zeros((self.nx*self.ny, 3), device=self.device, dtype=self.dtype)
        area_weights = weights * self.node_area[:, None]
        field[:, 2].scatter_add_(0, indices.flatten(), area_weights.flatten())
        for axis in range(2):
            field[:, axis].scatter_add_(0, indices.flatten(), (area_weights * velocity[:, axis, None]).flatten())
        coverage = (field[:, 2] / self.cell_area).clamp(0, 1)
        field[:, :2] /= field[:, 2, None].clamp_min(1e-20)
        field[:, :2] *= coverage[:, None] * self.shear_fraction
        return field[:, :2].reshape(self.ny, self.nx, 2), velocity

    def _wick(self, dt, node_momentum):
        """Conservative saturation diffusion, carrying donor-velocity momentum.

        ``node_momentum`` is total dry+wet cloth momentum immediately before
        wicking. Uptake from the film adds mass but zero momentum (stationary-film
        approximation). The returned/accumulated momentum change is available for
        the caller's mass update: v_new=(m_old*v_old+momentum_transfer)/m_new.
        """
        self.momentum_transfer.zero_()
        if self.wicking_diffusivity == 0:
            self.last_wicked_mass = 0.0
            self.last_wick_substeps = 0
            return
        substeps = max(1, math.ceil(dt*self.wick_max_rate/0.45))
        if substeps > 200:
            raise RuntimeError(f"Cotton wicking requires {substeps} substeps; reduce dt")
        left, right = self.wick_edges[:, 0], self.wick_edges[:, 1]
        total_wicked = torch.zeros((), device=self.device, dtype=self.dtype)
        for _ in range(substeps):
            # Positive signed mass travels left -> right, negative right -> left.
            proposed = self.wick_conductance * (self.wetness[left]-self.wetness[right]) * (dt/substeps)
            positive, negative = proposed.clamp_min(0), (-proposed).clamp_min(0)
            outgoing, incoming = torch.zeros_like(self.absorbed_mass), torch.zeros_like(self.absorbed_mass)
            outgoing.scatter_add_(0, left, positive)
            outgoing.scatter_add_(0, right, negative)
            incoming.scatter_add_(0, right, positive)
            incoming.scatter_add_(0, left, negative)
            donor_scale = (self.absorbed_mass/outgoing.clamp_min(1e-30)).clamp(0, 1)
            receiver_scale = ((self.capacity-self.absorbed_mass).clamp_min(0)/incoming.clamp_min(1e-30)).clamp(0, 1)
            edge_scale = torch.where(proposed >= 0,
                torch.minimum(donor_scale[left], receiver_scale[right]),
                torch.minimum(donor_scale[right], receiver_scale[left]))
            flux = proposed * edge_scale
            node_velocity = node_momentum / (self.dry_node_mass+self.absorbed_mass).clamp_min(1e-20)[:, None]
            donor_velocity = torch.where((flux >= 0)[:, None], node_velocity[left], node_velocity[right])
            momentum_flux = flux[:, None] * donor_velocity
            delta_mass = torch.zeros_like(self.absorbed_mass)
            delta_mass.scatter_add_(0, left, -flux)
            delta_mass.scatter_add_(0, right, flux)
            delta_momentum = torch.zeros_like(node_momentum)
            delta_momentum.index_add_(0, left, -momentum_flux)
            delta_momentum.index_add_(0, right, momentum_flux)
            self.absorbed_mass += delta_mass
            node_momentum += delta_momentum
            self.momentum_transfer += delta_momentum
            total_wicked += flux.abs().sum()
        self.last_wicked_mass = float(total_wicked.item())
        self.last_wick_substeps = substeps

    def _transport(self, dt, velocity):
        h = self.height
        padded = F.pad(h[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        lap = (padded[1:-1, 2:] - 2*h + padded[1:-1, :-2]) / self.dx**2 + (padded[2:, 1:-1] - 2*h + padded[:-2, 1:-1]) / self.dy**2
        pressure = self.rho * 9.81 * h - self.gamma * lap
        mobility_x = (h[:, :-1]**3 + h[:, 1:]**3) / (6*self.mu)
        mobility_y = (h[:-1, :]**3 + h[1:, :]**3) / (6*self.mu)
        ux = (velocity[:, :-1, 0] + velocity[:, 1:, 0]) / 2
        uy = (velocity[:-1, :, 1] + velocity[1:, :, 1]) / 2
        # Signed volume transfers, with no external face at domain boundaries.
        tx = (-mobility_x * (pressure[:, 1:]-pressure[:, :-1])/self.dx + ux*torch.where(ux >= 0, h[:, :-1], h[:, 1:])) * self.dy * dt
        ty = (-mobility_y * (pressure[1:, :]-pressure[:-1, :])/self.dy + uy*torch.where(uy >= 0, h[:-1, :], h[1:, :])) * self.dx * dt
        outgoing = torch.zeros_like(h)
        outgoing[:, :-1] += tx.clamp_min(0)
        outgoing[:, 1:] += (-tx).clamp_min(0)
        outgoing[:-1, :] += ty.clamp_min(0)
        outgoing[1:, :] += (-ty).clamp_min(0)
        scale = (h*self.cell_area / outgoing.clamp_min(1e-30)).clamp_max(1)
        tx *= torch.where(tx >= 0, scale[:, :-1], scale[:, 1:])
        ty *= torch.where(ty >= 0, scale[:-1, :], scale[1:, :])
        volume = h * self.cell_area
        volume[:, :-1] -= tx
        volume[:, 1:] += tx
        volume[:-1, :] -= ty
        volume[1:, :] += ty
        self.height = volume.clamp_min(0) / self.cell_area

    @torch.no_grad()
    def step(self, dt, cloth_positions, cloth_velocities=None, return_diagnostics=False):
        if dt <= 0:
            raise ValueError("dt must be positive")
        pos = torch.as_tensor(cloth_positions, device=self.device, dtype=self.dtype)
        if pos.shape != (len(self.node_area), 3):
            raise ValueError(f"Expected cloth positions {(len(self.node_area),3)}, got {tuple(pos.shape)}")
        indices, weights = self._links(pos)
        velocity, node_velocity = self._cloth_velocity(pos, dt, indices, weights, cloth_velocities)
        # Incoming film is approximated as stationary, so its uptake adds no
        # momentum. Preserve pre-uptake cloth momentum for the mechanical caller.
        node_momentum = (self.dry_node_mass+self.absorbed_mass)[:, None] * node_velocity
        # Conservative explicit stability restriction for gravity/capillarity/advection.
        inv_l2 = 1/self.dx**2 + 1/self.dy**2
        hmax = float(self.height.max().item())
        umax = float(velocity.abs().max().item())
        rate = hmax**3/(3*self.mu) * (self.rho*9.81*inv_l2 + 4*self.gamma*inv_l2**2) + umax*(1/self.dx + 1/self.dy)
        substeps = max(1, math.ceil(dt * rate / 0.15))
        if substeps > 200:
            raise RuntimeError(f"Fluid timestep requires {substeps} substeps; reduce dt / inspect cloth speed")
        for _ in range(substeps):
            self._transport(dt/substeps, velocity)
        # Re-evaluate contact with the updated local liquid height.
        indices, weights = self._links(pos)
        unsaturated = (1 - self.wetness).clamp(0, 1)
        demand = self.rho * self.node_area[:, None] * self.uptake_speed * dt * unsaturated[:, None] * weights
        remaining_capacity = (self.capacity - self.absorbed_mass).clamp_min(0)
        demand *= (remaining_capacity / demand.sum(-1).clamp_min(1e-30)).clamp_max(1)[:, None]
        cell_demand = torch.zeros(self.nx*self.ny, device=self.device, dtype=self.dtype)
        cell_demand.scatter_add_(0, indices.flatten(), demand.flatten())
        free_mass = self.height.flatten() * (self.rho*self.cell_area)
        cell_scale = (free_mass / cell_demand.clamp_min(1e-30)).clamp_max(1)
        transferred = demand * cell_scale[indices]
        cell_transfer = torch.zeros_like(free_mass)
        cell_transfer.scatter_add_(0, indices.flatten(), transferred.flatten())
        self.height = ((free_mass - cell_transfer).clamp_min(0) / (self.rho*self.cell_area)).reshape(self.ny, self.nx)
        self.absorbed_mass += transferred.sum(-1)
        self._wick(dt, node_momentum)
        self.last_contact_area = float((self.node_area * weights.sum(-1).clamp_max(1)).sum().item())
        self.last_uptake = float(transferred.sum().item())
        self.last_substeps = substeps
        self.elapsed += dt
        self.steps += 1
        return self.diagnostics() if return_diagnostics else None

    def diagnostics(self):
        free = float((self.height.sum()*self.cell_area*self.rho).item())
        absorbed = float(self.absorbed_mass.sum().item())
        return dict(time_s=self.elapsed, steps=self.steps, initial_liquid_mass_kg=self.initial_mass,
            free_liquid_mass_kg=free, absorbed_liquid_mass_kg=absorbed,
            total_liquid_mass_kg=free+absorbed,
            relative_mass_error=(free+absorbed-self.initial_mass)/self.initial_mass,
            residual_fraction=free/self.initial_mass,
            dry_cloth_mass_kg=float(self.dry_node_mass.sum().item()),
            cloth_capacity_kg=float(self.capacity.sum().item()),
            max_saturation=float(self.wetness.max().item()),
            contact_area_m2=self.last_contact_area, last_uptake_kg=self.last_uptake,
            last_internal_wicked_mass_kg=self.last_wicked_mass,
            last_wicking_substeps=self.last_wick_substeps,
            internal_momentum_balance_kg_m_s=float(self.momentum_transfer.sum(0).norm().item()),
            min_height_m=float(self.height.min().item()), max_height_m=float(self.height.max().item()),
            last_fluid_substeps=self.last_substeps, device=str(self.device))

    def film_vertices(self):
        """Flat-grid CUDA vertices; position is the true computed liquid surface."""
        return torch.cat((self.xy.reshape(-1, 2), (self.height.flatten()+self.table_z)[:, None]), -1)

    def render_state(self, height_threshold=2e-7):
        """CPU mesh arrays and exact film heights. Threshold only affects rendering."""
        vertices = self.film_vertices().detach().cpu().numpy()
        h = self.height.detach().cpu().numpy().reshape(-1)
        # Require at least one vertex with visible liquid; mass itself is never removed.
        faces = self.film_faces[np.max(h[self.film_faces], axis=1) > height_threshold]
        return dict(vertices=vertices, faces=faces, height=h, wetness=self.wetness.detach().cpu().numpy())


def self_test(asset_directory, device="cuda"):
    cloth = trimesh.load(Path(asset_directory)/"cotton_cloth.obj", force="mesh", process=False)
    rest = np.asarray(cloth.vertices).copy()
    common = dict(asset_path=Path(asset_directory)/"soy_puddle.obj", cloth_rest_vertices=rest,
        cloth_faces=np.asarray(cloth.faces), device=device)
    model = ThinFilmAbsorption(**common)
    far = rest + [0, .04, .82]
    for _ in range(100):
        model.step(.01, far)
    no_contact = model.diagnostics()
    assert no_contact["absorbed_liquid_mass_kg"] == 0, no_contact
    assert abs(no_contact["relative_mass_error"]) < 1e-10, no_contact
    assert no_contact["min_height_m"] >= 0, no_contact
    # A resting/reciprocating cloth must transfer liquid to material-local capacity.
    for i in range(800):
        pos = rest + [0.065*np.sin(i*.01*2), .04+0.035*np.sin(i*.01*1.4), .752]
        model.step(.01, pos)
    wiping = model.diagnostics()
    assert wiping["absorbed_liquid_mass_kg"] > .5 * model.initial_mass, wiping
    assert abs(wiping["relative_mass_error"]) < 1e-10, wiping
    assert wiping["max_saturation"] <= 1+1e-12, wiping
    # Lifting must cease uptake immediately, while carrying all absorbed liquid.
    before = wiping["absorbed_liquid_mass_kg"]
    for _ in range(20):
        model.step(.01, far)
    lifted = model.diagnostics()
    assert abs(lifted["absorbed_liquid_mass_kg"]-before) < 1e-14, lifted
    # Capacity cannot be exceeded even with artificially aggressive uptake.
    limited = ThinFilmAbsorption(**common, capacity_ratio=.01, uptake_speed=1)
    for _ in range(30):
        limited.step(.01, rest+[0,.04,.752])
    saturated = limited.diagnostics()
    assert saturated["absorbed_liquid_mass_kg"] <= saturated["cloth_capacity_kg"]*(1+1e-10), saturated
    assert saturated["free_liquid_mass_kg"] > .9*limited.initial_mass, saturated
    assert abs(saturated["relative_mass_error"]) < 1e-10, saturated
    # Isolate internal wicking: place existing liquid in a small material patch,
    # then hold the cloth above the table so there can be no new uptake.
    wick = ThinFilmAbsorption(**common)
    wick.height.zero_()
    patch = torch.as_tensor((np.abs(rest[:, 0]) < .025) & (np.abs(rest[:, 1]) < .025), device=wick.device)
    wick.absorbed_mass[patch] = wick.capacity[patch] * .85
    wick.initial_mass = float(wick.absorbed_mass.sum().item())
    initial_variance = float(torch.sum(wick.capacity*(wick.wetness-wick.absorbed_mass.sum()/wick.capacity.sum())**2).item())
    initially_dry = wick.absorbed_mass == 0
    # Heterogeneous velocities exercise donor-velocity momentum transport. The
    # same mass/momentum update used by Genesis' caller is reproduced here.
    velocity = torch.as_tensor(np.stack((.05*np.sin(rest[:, 1]/.08), .03*np.cos(rest[:, 0]/.10), .01*np.sin(rest[:, 0]/.07)), axis=1), device=wick.device, dtype=wick.dtype)
    initial_momentum = ((wick.dry_node_mass+wick.absorbed_mass)[:, None]*velocity).sum(0)
    max_pair_momentum_error = 0.0
    for _ in range(100):
        old_mass = wick.dry_node_mass+wick.absorbed_mass.clone()
        wick.step(.01, far, cloth_velocities=velocity)
        new_mass = wick.dry_node_mass+wick.absorbed_mass
        velocity = (old_mass[:, None]*velocity+wick.momentum_transfer)/new_mass[:, None]
        max_pair_momentum_error = max(max_pair_momentum_error, float(wick.momentum_transfer.sum(0).norm().item()))
        assert float(wick.absorbed_mass.min().item()) >= -1e-18
        assert float(wick.wetness.max().item()) <= 1+1e-12
        assert wick.last_uptake == 0
    final_momentum = ((wick.dry_node_mass+wick.absorbed_mass)[:, None]*velocity).sum(0)
    final_variance = float(torch.sum(wick.capacity*(wick.wetness-wick.absorbed_mass.sum()/wick.capacity.sum())**2).item())
    isolated_wicking = wick.diagnostics()
    isolated_wicking.update(initial_weighted_saturation_variance=initial_variance,
        final_weighted_saturation_variance=final_variance,
        mass_spread_to_previously_dry_nodes_kg=float(wick.absorbed_mass[initially_dry].sum().item()),
        total_momentum_error_kg_m_s=float((final_momentum-initial_momentum).norm().item()),
        max_internal_momentum_error_kg_m_s=max_pair_momentum_error)
    assert abs(isolated_wicking["relative_mass_error"]) < 1e-10, isolated_wicking
    assert isolated_wicking["mass_spread_to_previously_dry_nodes_kg"] > 0, isolated_wicking
    assert final_variance < initial_variance, isolated_wicking
    assert isolated_wicking["total_momentum_error_kg_m_s"] < 1e-14, isolated_wicking
    assert max_pair_momentum_error < 1e-14, isolated_wicking
    # A coarse wicking interval must still respect both bounds through substeps.
    wick._wick(.5, (wick.dry_node_mass+wick.absorbed_mass)[:, None]*velocity)
    assert float(wick.absorbed_mass.min().item()) >= -1e-18
    assert float(wick.wetness.max().item()) <= 1+1e-12
    assert abs(wick.diagnostics()["relative_mass_error"]) < 1e-10
    result = dict(passed=True, no_contact=no_contact, wiping=wiping, lifted=lifted,
        limited_capacity=saturated, isolated_wicking=isolated_wicking)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--assets", default=str(Path(__file__).parent/"assets"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.self_test:
        self_test(args.assets, args.device)
    else:
        parser.print_help()
