"""mik-gingos: a Navier-Stokes fluid simulator in one file.

The incompressible Navier-Stokes equations

    du/dt + (u . grad) u = -grad(p) / rho + nu * laplacian(u) + f     (momentum)
    div(u) = 0                                                        (mass)

are solved twice: in Python with NumPy on the server, and in JavaScript in
the browser so you can stir the fluid in real time. Both use Jos Stam's
"Stable Fluids" operator splitting:

    add forces -> diffuse -> project -> advect -> project

The web page is styled as a Swiss + brutalist mix.

Install:          pip install flask numpy
Run the app:      python3 main.py               then open http://127.0.0.1:5000
Run the tests:    python3 main.py --test
Render a frame:   python3 main.py render --scenario karman --steps 300 --out karman.png
List scenarios:   python3 main.py scenarios
Benchmark:        python3 main.py bench
"""

import argparse
import base64
import functools
import json
import math
import struct
import sys
import time
import unittest
import zlib
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)

BOUNDARY_MODES = ("box", "periodic", "tunnel")
ADVECTION_SCHEMES = ("semi-lagrangian", "maccormack")
FIELDS = ("dye", "speed", "vorticity", "pressure", "temperature")

MIN_N, MAX_N, DEFAULT_N = 16, 128, 64
MAX_STEPS, DEFAULT_STEPS = 600, 200
# Server runs are capped at this many cell updates (cells x steps) so one
# request cannot tie the server up for minutes.
MAX_WORK = 64 * 64 * 400


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Boundary:
    """What happens at the edges of the domain.

    box       solid walls on all four sides; ``lid`` drags the top wall sideways
    periodic  fluid leaving one side comes back on the opposite side
    tunnel    inflow at ``inflow`` speed on the left, open outflow on the right,
              walls at top and bottom
    """

    mode: str = "box"
    inflow: float = 0.0
    lid: float = 0.0

    def __post_init__(self):
        if self.mode not in BOUNDARY_MODES:
            raise ValueError(f"unknown boundary mode {self.mode!r}; pick one of {BOUNDARY_MODES}")


@dataclass
class FluidParams:
    """Physical constants and numerical settings for a simulation."""

    viscosity: float = 0.0       # nu, kinematic viscosity
    diffusion: float = 0.0       # kappa, how fast dye and heat spread out
    dissipation: float = 0.0     # dye fades at this rate (1/s)
    buoyancy: float = 0.0        # upward acceleration per unit temperature
    weight: float = 0.0          # downward acceleration per unit dye
    cooling: float = 0.0         # temperature relaxes to zero at this rate (1/s)
    vorticity: float = 0.0       # epsilon, vorticity-confinement strength
    pressure_iters: int = 40     # red-black SOR sweeps for the pressure solve
    diffusion_iters: int = 20    # red-black Gauss-Seidel sweeps for diffusion
    sor: float = 1.5             # over-relaxation factor for the pressure solve
    advection: str = "semi-lagrangian"

    def __post_init__(self):
        if self.advection not in ADVECTION_SCHEMES:
            raise ValueError(f"unknown advection scheme {self.advection!r}")
        if not 0.0 < self.sor < 2.0:
            raise ValueError("sor must be between 0 and 2")
        for name in ("viscosity", "diffusion", "dissipation", "cooling", "vorticity"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.pressure_iters < 1 or self.diffusion_iters < 1:
            raise ValueError("iteration counts must be at least 1")


class FluidSolver:
    """Incompressible 2D Navier-Stokes on a collocated grid.

    Every field has one layer of ghost cells, so arrays have shape
    ``(ny + 2, nx + 2)`` and the fluid lives in ``[1:-1, 1:-1]``. Rows run
    top to bottom (like an image), so a positive ``v`` points down.
    """

    def __init__(self, nx, ny, width=1.0, boundary=None, params=None):
        if nx < 4 or ny < 4:
            raise ValueError("the grid must be at least 4 x 4 cells")
        if width <= 0:
            raise ValueError("width must be positive")
        self.boundary = boundary or Boundary()
        if self.boundary.mode == "periodic" and (nx % 2 or ny % 2):
            raise ValueError("periodic grids need an even number of cells in each direction")
        self.params = params or FluidParams()
        self.nx, self.ny = nx, ny
        self.h = width / nx
        self.width, self.height = width, self.h * ny

        shape = (ny + 2, nx + 2)
        self.u = np.zeros(shape)        # velocity, x component
        self.v = np.zeros(shape)        # velocity, y component (down is positive)
        self.p = np.zeros(shape)        # pressure (scaled by dt / rho)
        self.dye = np.zeros(shape)      # passive tracer that makes the flow visible
        self.temp = np.zeros(shape)     # temperature, drives buoyancy
        self.solid = np.zeros(shape, dtype=bool)
        self.time = 0.0
        self.steps = 0
        self.emitters: List[Callable[["FluidSolver", float], None]] = []

        jj, ii = np.mgrid[0:ny + 2, 0:nx + 2]
        self._I = ii[1:-1, 1:-1].astype(float)
        self._J = jj[1:-1, 1:-1].astype(float)
        # Physical position of every cell centre, ghost cells included.
        self.X = (ii - 0.5) * self.h
        self.Y = (jj - 0.5) * self.h
        # Red-black colouring as (row offset, column offset) sub-grids of stride 2.
        self._colours = (((0, 0), (1, 1)), ((0, 1), (1, 0)))
        self._update_masks()

    # -- obstacles ----------------------------------------------------------

    def _update_masks(self):
        s = self.solid
        s[0, :] = s[-1, :] = False
        s[:, 0] = s[:, -1] = False
        self._fluid = ~s[1:-1, 1:-1]
        self._sE = s[1:-1, 2:].copy()
        self._sW = s[1:-1, :-2].copy()
        self._sS = s[2:, 1:-1].copy()
        self._sN = s[:-2, 1:-1].copy()
        self._wE, self._wW, self._wS, self._wN = (
            (~m).astype(float) for m in (self._sE, self._sW, self._sS, self._sN)
        )
        count = self._wE + self._wW + self._wS + self._wN
        self._count = np.where(count > 0, count, 1.0)
        self._fluid_f = self._fluid.astype(float)

    def _after_obstacles(self):
        self._update_masks()
        for f in (self.u, self.v, self.dye, self.temp):
            f[self.solid] = 0.0

    def add_circle(self, cx, cy, r):
        """Make every cell whose centre is within ``r`` of (cx, cy) solid."""
        self.solid |= (self.X - cx) ** 2 + (self.Y - cy) ** 2 <= r * r
        self._after_obstacles()

    def add_rect(self, x0, y0, x1, y1):
        """Make every cell whose centre is inside the rectangle solid."""
        self.solid |= (self.X >= x0) & (self.X <= x1) & (self.Y >= y0) & (self.Y <= y1)
        self._after_obstacles()

    def clear_obstacles(self):
        self.solid[:] = False
        self._update_masks()

    # -- boundaries ---------------------------------------------------------

    def set_bnd(self, f, kind):
        """Fill the ghost cells of ``f``.

        ``kind`` is "u" or "v" for the velocity components, "p" for pressure
        and "s" for any other scalar (dye, temperature).
        """
        mode = self.boundary.mode
        if mode == "periodic":
            f[0, :] = f[-2, :]
            f[-1, :] = f[1, :]
            f[:, 0] = f[:, -2]
            f[:, -1] = f[:, 1]
            return f

        # Top and bottom are walls in both "box" and "tunnel".
        if kind == "v":
            f[0, 1:-1] = -f[1, 1:-1]
            f[-1, 1:-1] = -f[-2, 1:-1]
        else:
            f[0, 1:-1] = f[1, 1:-1]
            f[-1, 1:-1] = f[-2, 1:-1]
            if kind == "u" and mode == "box" and self.boundary.lid:
                f[0, 1:-1] = 2.0 * self.boundary.lid - f[1, 1:-1]

        if mode == "box":
            sign = -1.0 if kind == "u" else 1.0
            f[1:-1, 0] = sign * f[1:-1, 1]
            f[1:-1, -1] = sign * f[1:-1, -2]
        else:
            if kind == "u":
                f[1:-1, 0] = 2.0 * self.boundary.inflow - f[1:-1, 1]
            elif kind == "p":
                f[1:-1, 0] = f[1:-1, 1]
            else:
                f[1:-1, 0] = -f[1:-1, 1]
            # Open outflow: zero gradient for everything, pressure pinned to 0.
            f[1:-1, -1] = -f[1:-1, -2] if kind == "p" else f[1:-1, -2]

        f[0, 0] = 0.5 * (f[1, 0] + f[0, 1])
        f[0, -1] = 0.5 * (f[1, -1] + f[0, -2])
        f[-1, 0] = 0.5 * (f[-2, 0] + f[-1, 1])
        f[-1, -1] = 0.5 * (f[-2, -1] + f[-1, -2])
        return f

    def _enforce_velocity(self):
        self.set_bnd(self.u, "u")
        self.set_bnd(self.v, "v")
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0

    # -- advection ----------------------------------------------------------

    def _departure(self, u, v, dt):
        """Trace every cell centre back along the flow for ``dt`` seconds."""
        x = self._I - dt * u[1:-1, 1:-1] / self.h
        y = self._J - dt * v[1:-1, 1:-1] / self.h
        if self.boundary.mode == "periodic":
            x = np.mod(x - 1.0, self.nx) + 1.0
            y = np.mod(y - 1.0, self.ny) + 1.0
        else:
            np.clip(x, 0.5, self.nx + 0.5, out=x)
            np.clip(y, 0.5, self.ny + 0.5, out=y)
        return x, y

    def _sample(self, f, x, y, bounds=False):
        """Bilinear interpolation of ``f`` at fractional cell indices."""
        i0 = np.clip(np.floor(x).astype(np.intp), 0, self.nx)
        j0 = np.clip(np.floor(y).astype(np.intp), 0, self.ny)
        sx = x - i0
        sy = y - j0
        a = f[j0, i0]
        b = f[j0, i0 + 1]
        c = f[j0 + 1, i0]
        d = f[j0 + 1, i0 + 1]
        value = (1 - sy) * ((1 - sx) * a + sx * b) + sy * ((1 - sx) * c + sx * d)
        if not bounds:
            return value
        lo = np.minimum(np.minimum(a, b), np.minimum(c, d))
        hi = np.maximum(np.maximum(a, b), np.maximum(c, d))
        return value, lo, hi

    def advect(self, fields, u, v, dt):
        """Carry each ``(field, kind)`` pair along the velocity (u, v).

        Semi-Lagrangian advection looks up where each cell's fluid came from
        and interpolates there, which is stable for any time step. MacCormack
        adds a backward pass to cancel most of the smoothing error, and clamps
        the result so it never creates new peaks.
        """
        maccormack = self.params.advection == "maccormack"
        x, y = self._departure(u, v, dt)
        if maccormack:
            xb, yb = self._departure(u, v, -dt)
        out = []
        for f, kind in fields:
            self.set_bnd(f, kind)
            g = f.copy()
            if maccormack:
                forward, lo, hi = self._sample(f, x, y, bounds=True)
                g[1:-1, 1:-1] = forward
                self.set_bnd(g, kind)
                back = self._sample(g, xb, yb)
                corrected = forward + 0.5 * (f[1:-1, 1:-1] - back)
                g[1:-1, 1:-1] = np.clip(corrected, lo, hi)
            else:
                g[1:-1, 1:-1] = self._sample(f, x, y)
            self.set_bnd(g, kind)
            out.append(g)
        return out

    # -- linear solvers -----------------------------------------------------

    def _relax(self, x, rhs, a, c, kind, iters, omega=1.0, weighted=False):
        """Solve ``c * x - a * (sum of 4 neighbours) = rhs`` in place.

        Uses red-black Gauss-Seidel (with over-relaxation ``omega``), which
        vectorises well: every red cell only depends on black cells and the
        other way round. Each colour is two interleaved sub-grids with stride
        2, updated through slices so nothing is copied. With ``weighted`` set,
        solid neighbours are skipped and solid cells are left alone, which
        gives the pressure a zero-gradient condition at obstacles.
        """
        ny, nx = self.ny, self.nx
        for _ in range(iters):
            for colour in self._colours:
                for ra, cb in colour:
                    sub = (slice(ra, None, 2), slice(cb, None, 2))
                    rows = slice(1 + ra, ny + 1, 2)
                    cols = slice(1 + cb, nx + 1, 2)
                    centre = x[rows, cols]
                    east = x[rows, 2 + cb:nx + 2:2]
                    west = x[rows, cb:nx:2]
                    south = x[2 + ra:ny + 2:2, cols]
                    north = x[ra:ny:2, cols]
                    if weighted:
                        nbr = (east * self._wE[sub] + west * self._wW[sub]
                               + south * self._wS[sub] + north * self._wN[sub])
                    else:
                        nbr = east + west + south + north
                    cc = c if np.isscalar(c) else c[sub]
                    change = omega * ((rhs[sub] + a * nbr) / cc - centre)
                    if weighted:
                        change *= self._fluid_f[sub]
                    centre += change
                self.set_bnd(x, kind)
        return x

    def diffuse(self, f, rate, dt, kind):
        """Implicit diffusion: solve (I - rate * dt * laplacian) f_new = f."""
        if rate <= 0:
            return f
        a = rate * dt / self.h ** 2
        rhs = f[1:-1, 1:-1].copy()
        return self._relax(f, rhs, a, 1.0 + 4.0 * a, kind, self.params.diffusion_iters)

    # -- projection ---------------------------------------------------------

    def divergence(self):
        """Central-difference divergence of the velocity, zero inside solids."""
        self.set_bnd(self.u, "u")
        self.set_bnd(self.v, "v")
        u, v = self.u, self.v
        div = (u[1:-1, 2:] - u[1:-1, :-2] + v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * self.h)
        div[~self._fluid] = 0.0
        return div

    def project(self):
        """Make the velocity divergence-free (Helmholtz-Hodge projection).

        Solve the Poisson equation laplacian(p) = div(u), then subtract
        grad(p) from u. What is left has no sources or sinks, which is what
        the incompressibility equation div(u) = 0 asks for.
        """
        h = self.h
        rhs = -h * h * self.divergence()
        self._relax(self.p, rhs, 1.0, self._count, "p", self.params.pressure_iters,
                    omega=self.params.sor, weighted=True)
        if self.boundary.mode != "tunnel":
            # With walls or periodic edges p is only defined up to a constant.
            inner = self.p[1:-1, 1:-1]
            inner -= inner[self._fluid].mean()
            self.set_bnd(self.p, "p")
        p = self.p
        pc = p[1:-1, 1:-1]
        pE = np.where(self._sE, pc, p[1:-1, 2:])
        pW = np.where(self._sW, pc, p[1:-1, :-2])
        pS = np.where(self._sS, pc, p[2:, 1:-1])
        pN = np.where(self._sN, pc, p[:-2, 1:-1])
        self.u[1:-1, 1:-1] -= (pE - pW) / (2.0 * h)
        self.v[1:-1, 1:-1] -= (pS - pN) / (2.0 * h)
        self._enforce_velocity()

    # -- forces -------------------------------------------------------------

    def vorticity(self):
        """Curl of the velocity, dv/dx - du/dy, for every fluid cell."""
        self.set_bnd(self.u, "u")
        self.set_bnd(self.v, "v")
        u, v = self.u, self.v
        return (v[1:-1, 2:] - v[1:-1, :-2] - u[2:, 1:-1] + u[:-2, 1:-1]) / (2.0 * self.h)

    def _confine(self, dt):
        """Vorticity confinement: put back small swirls that the grid smears out."""
        eps = self.params.vorticity
        if eps <= 0:
            return
        w = self.vorticity()
        pad = "wrap" if self.boundary.mode == "periodic" else "edge"
        aw = np.pad(np.abs(w), 1, mode=pad)
        gx = (aw[1:-1, 2:] - aw[1:-1, :-2]) / (2.0 * self.h)
        gy = (aw[2:, 1:-1] - aw[:-2, 1:-1]) / (2.0 * self.h)
        mag = np.sqrt(gx * gx + gy * gy) + 1e-12
        scale = dt * eps * self.h * w * self._fluid / mag
        self.u[1:-1, 1:-1] += scale * gy
        self.v[1:-1, 1:-1] -= scale * gx

    def _apply_forces(self, dt):
        P = self.params
        if P.buoyancy or P.weight:
            self.v[1:-1, 1:-1] += dt * (P.weight * self.dye[1:-1, 1:-1]
                                        - P.buoyancy * self.temp[1:-1, 1:-1])
        self._confine(dt)

    # -- time stepping ------------------------------------------------------

    def step(self, dt):
        """Advance the simulation by ``dt`` seconds."""
        if not (isinstance(dt, (int, float)) and math.isfinite(dt) and dt > 0):
            raise ValueError("dt must be a positive number")
        P = self.params
        for emit in self.emitters:
            emit(self, dt)

        # Velocity: forces, viscosity, projection, self-advection, projection.
        self._apply_forces(dt)
        if P.viscosity > 0:
            self.diffuse(self.u, P.viscosity, dt, "u")
            self.diffuse(self.v, P.viscosity, dt, "v")
        self._enforce_velocity()
        self.project()
        self.u, self.v = self.advect([(self.u, "u"), (self.v, "v")], self.u, self.v, dt)
        self._enforce_velocity()
        self.project()

        # Scalars ride along with the new, divergence-free velocity.
        self.temp, self.dye = self.advect([(self.temp, "s"), (self.dye, "s")], self.u, self.v, dt)
        if P.diffusion > 0:
            self.diffuse(self.dye, P.diffusion, dt, "s")
            self.diffuse(self.temp, P.diffusion, dt, "s")
        if P.cooling:
            self.temp *= 1.0 / (1.0 + P.cooling * dt)
        if P.dissipation:
            self.dye *= 1.0 / (1.0 + P.dissipation * dt)
        self.dye[self.solid] = 0.0
        self.temp[self.solid] = 0.0

        self.time += dt
        self.steps += 1

    def max_speed(self):
        u = self.u[1:-1, 1:-1]
        v = self.v[1:-1, 1:-1]
        return float(np.sqrt(np.max(u * u + v * v)))

    def stable_dt(self, cfl=1.0):
        """Time step that moves the fastest fluid ``cfl`` cells per step."""
        speed = max(self.max_speed(), abs(self.boundary.inflow), abs(self.boundary.lid))
        return math.inf if speed < 1e-12 else cfl * self.h / speed

    def advance(self, duration, cfl=1.0, max_dt=0.05):
        """Run for ``duration`` seconds in CFL-limited sub-steps; return the step count."""
        elapsed, count = 0.0, 0
        while duration - elapsed > 1e-12:
            dt = min(self.stable_dt(cfl), max_dt, duration - elapsed)
            self.step(dt)
            elapsed += dt
            count += 1
        return count

    # -- measurements -------------------------------------------------------

    def kinetic_energy(self):
        u = self.u[1:-1, 1:-1]
        v = self.v[1:-1, 1:-1]
        return float(0.5 * np.sum((u * u + v * v)[self._fluid]) * self.h ** 2)

    def enstrophy(self):
        w = self.vorticity()
        return float(0.5 * np.sum((w * w)[self._fluid]) * self.h ** 2)

    def dye_mass(self):
        return float(np.sum(self.dye[1:-1, 1:-1][self._fluid]) * self.h ** 2)

    def diagnostics(self):
        div = np.abs(self.divergence()[self._fluid])
        return {
            "time": self.time,
            "steps": self.steps,
            "grid": [self.nx, self.ny],
            "kinetic_energy": self.kinetic_energy(),
            "enstrophy": self.enstrophy(),
            "max_speed": self.max_speed(),
            "max_divergence": float(div.max()) if div.size else 0.0,
            "rms_divergence": float(np.sqrt(np.mean(div * div))) if div.size else 0.0,
            "dye_mass": self.dye_mass(),
        }

    def field(self, name):
        """One of FIELDS as a ``(ny, nx)`` array."""
        if name == "dye":
            return self.dye[1:-1, 1:-1].copy()
        if name == "temperature":
            return self.temp[1:-1, 1:-1].copy()
        if name == "pressure":
            return self.p[1:-1, 1:-1].copy()
        if name == "speed":
            u = self.u[1:-1, 1:-1]
            v = self.v[1:-1, 1:-1]
            return np.sqrt(u * u + v * v)
        if name == "vorticity":
            return self.vorticity()
        raise ValueError(f"unknown field {name!r}; pick one of {FIELDS}")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

TAYLOR_GREEN_NU = 0.05


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str
    aspect: int                      # cells across for every cell down
    dt: float                        # time step at 64 rows; scaled for other grids
    field: str                       # field shown by default
    build: Callable[[int], FluidSolver]

    def make(self, n):
        """Build a fresh solver that is ``n`` cells tall."""
        return self.build(n)

    def time_step(self, n):
        return self.dt * 64.0 / n


def _ellipse(s, cx, cy, rx, ry):
    return ((s.X - cx) / rx) ** 2 + ((s.Y - cy) / ry) ** 2 <= 1.0


def _build_plume(n):
    s = FluidSolver(n, n, 1.0, Boundary("box"),
                    FluidParams(buoyancy=6.0, cooling=0.3, dissipation=0.02, vorticity=1.5))
    source = _ellipse(s, 0.5, 0.9, 0.07, 0.025)

    def emit(sol, dt):
        sol.dye[source] = 1.0
        sol.temp[source] = 1.0
        sol.u[source] = 0.15 * math.sin(4.0 * sol.time)
        sol.v[source] = -0.4

    s.emitters.append(emit)
    return s


def _build_karman(n):
    s = FluidSolver(2 * n, n, 2.0, Boundary("tunnel", inflow=1.0),
                    FluidParams(viscosity=2e-4, pressure_iters=60, advection="maccormack"))
    s.add_circle(0.42, 0.5, 0.075)
    s.u[1:-1, 1:-1] = 1.0
    s.u[s.solid] = 0.0
    inlet = np.zeros_like(s.solid)
    inlet[1:-1, 1:3] = True
    inlet &= np.floor(s.Y * 12.0) % 2 == 0
    kick = _ellipse(s, 0.62, 0.5, 0.05, 0.05)

    def emit(sol, dt):
        sol.dye[inlet] = 1.0
        if sol.time < 0.3:
            # A small push sideways breaks the symmetry so vortices start shedding.
            sol.v[kick] += 2.0 * dt

    s.emitters.append(emit)
    return s


def _build_cavity(n):
    s = FluidSolver(n, n, 1.0, Boundary("box", lid=1.0),
                    FluidParams(viscosity=0.002, advection="maccormack"))
    s.dye[:] = (np.floor(s.Y * 10.0) % 2 == 0).astype(float)
    return s


def _build_shear(n):
    s = FluidSolver(n, n, 1.0, Boundary("periodic"),
                    FluidParams(viscosity=3e-4, advection="maccormack"))
    width = 0.025
    band = 0.5 * (np.tanh((s.Y - 0.25) / width) - np.tanh((s.Y - 0.75) / width))
    s.u[:] = 2.0 * band - 1.0
    bump = np.exp(-((s.Y - 0.25) / 0.05) ** 2) + np.exp(-((s.Y - 0.75) / 0.05) ** 2)
    s.v[:] = 0.05 * np.sin(4.0 * math.pi * s.X) * bump
    s.dye[:] = band
    s.project()
    return s


def _build_jets(n):
    s = FluidSolver(n, n, 1.0, Boundary("box"),
                    FluidParams(dissipation=0.05, vorticity=1.0))
    left = _ellipse(s, 0.08, 0.46, 0.04, 0.04)
    right = _ellipse(s, 0.92, 0.54, 0.04, 0.04)

    def emit(sol, dt):
        sol.u[left] = 1.2
        sol.v[left] = 0.0
        sol.dye[left] = 1.0
        sol.u[right] = -1.2
        sol.v[right] = 0.0
        sol.dye[right] = 1.0
        sol.temp[right] = 1.0

    s.emitters.append(emit)
    return s


def _build_taylor_green(n):
    s = FluidSolver(n, n, 2.0 * math.pi, Boundary("periodic"),
                    FluidParams(viscosity=TAYLOR_GREEN_NU, advection="maccormack"))
    s.u[:] = np.sin(s.X) * np.cos(s.Y)
    s.v[:] = -np.cos(s.X) * np.sin(s.Y)
    s.dye[:] = (np.floor(s.X * 4.0 / math.pi) % 2 == 0).astype(float)
    return s


def taylor_green_energy(e0, t, nu=TAYLOR_GREEN_NU):
    """Exact kinetic energy of the Taylor-Green vortex at time ``t``."""
    return e0 * math.exp(-4.0 * nu * t)


SCENARIOS: Dict[str, Scenario] = {s.key: s for s in (
    Scenario("plume", "Smoke plume",
             "Hot smoke rises from a vent. Buoyancy lifts it, vorticity confinement keeps the curls sharp.",
             1, 0.01, "dye", _build_plume),
    Scenario("karman", "Kármán vortex street",
             "Wind flows past a cylinder. The wake rolls up into vortices that peel off one side, then the other.",
             2, 0.008, "vorticity", _build_karman),
    Scenario("cavity", "Lid-driven cavity",
             "A closed box whose top wall slides to the right. The classic benchmark for viscous flow solvers.",
             1, 0.01, "dye", _build_cavity),
    Scenario("shear", "Kelvin–Helmholtz",
             "Two layers slide past each other. A small ripple grows into rolling billows, like wind over water.",
             1, 0.006, "dye", _build_shear),
    Scenario("jets", "Colliding jets",
             "Two jets fire into a closed box, slightly offset, and tear each other into eddies.",
             1, 0.008, "dye", _build_jets),
    Scenario("taylor_green", "Taylor–Green vortex",
             "A grid of vortices that slowly dies out from viscosity. Its exact solution is known, so it checks the solver.",
             1, 0.05, "vorticity", _build_taylor_green),
)}


def scenario_config():
    """Everything the browser needs to rebuild each scenario's settings."""
    out = []
    for s in SCENARIOS.values():
        probe = s.make(16)
        out.append({
            "key": s.key,
            "title": s.title,
            "description": s.description,
            "aspect": s.aspect,
            "dt": s.dt,
            "field": s.field,
            "width": probe.width,
            "boundary": asdict(probe.boundary),
            "params": asdict(probe.params),
        })
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

PAPER = "#f2f0eb"
PALETTES = {
    "ink": [PAPER, "#ff2a00", "#000000"],
    "heat": ["#000000", "#ff2a00", "#ffd400", "#ffffff"],
    "split": ["#000000", PAPER, "#ff2a00"],
}
FIELD_PALETTE = {
    "dye": "ink",
    "speed": "heat",
    "vorticity": "split",
    "pressure": "split",
    "temperature": "heat",
}
DIVERGING = {"vorticity", "pressure"}


def _hex_rgb(color):
    return [int(color[k:k + 2], 16) for k in (1, 3, 5)]


@functools.lru_cache(maxsize=None)
def colormap(name, size=256):
    """A ``(size, 3)`` uint8 lookup table interpolated between palette stops."""
    stops = np.array([_hex_rgb(c) for c in PALETTES[name]], dtype=float)
    t = np.linspace(0.0, 1.0, size)
    pos = np.linspace(0.0, 1.0, len(stops))
    lut = np.stack([np.interp(t, pos, stops[:, k]) for k in range(3)], axis=1)
    lut = np.round(lut).astype(np.uint8)
    lut.flags.writeable = False
    return lut


def normalize(name, values):
    """Map a field to [0, 1] for colouring. Diverging fields centre on 0.5."""
    if name == "dye":
        lo, hi = 0.0, 1.0
    elif name in DIVERGING:
        m = float(np.percentile(np.abs(values), 99)) if values.size else 0.0
        lo, hi = -max(m, 1e-9), max(m, 1e-9)
    else:
        top = float(np.percentile(values, 99.5)) if values.size else 0.0
        lo, hi = 0.0, max(top, 1e-9)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def render(solver, name, scale=1):
    """Colour one field of the solver as an RGB image; obstacles are hatched."""
    if name not in FIELDS:
        raise ValueError(f"unknown field {name!r}; pick one of {FIELDS}")
    scale = max(1, int(scale))
    t = normalize(name, solver.field(name))
    rgb = colormap(FIELD_PALETTE[name])[(t * 255).astype(np.intp)]
    solid = solver.solid[1:-1, 1:-1]
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        solid = np.repeat(np.repeat(solid, scale, axis=0), scale, axis=1)
    if solid.any():
        yy, xx = np.indices(solid.shape)
        stripe = ((xx + yy) // 3) % 2 == 0
        rgb[solid & stripe] = 0
        rgb[solid & ~stripe] = _hex_rgb(PAPER)
    return rgb


def encode_png(rgb):
    """Encode an ``(h, w, 3)`` uint8 array as a PNG file, no Pillow needed."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an (height, width, 3) array")
    height, width, _ = rgb.shape
    rows = np.zeros((height, width * 3 + 1), dtype=np.uint8)   # filter byte 0 per row
    rows[:, 1:] = rgb.reshape(height, width * 3)

    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)) + chunk(b"IEND", b""))


def decode_png(data):
    """Decode a PNG written by ``encode_png`` back into an array (used by tests)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos, idat, width = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if zlib.crc32(tag + body) & 0xFFFFFFFF != crc:
            raise ValueError(f"bad CRC in {tag!r} chunk")
        if tag == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    if width is None:
        raise ValueError("missing IHDR chunk")
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, width * 3 + 1)
    return raw[:, 1:].reshape(height, width, 3)


# ---------------------------------------------------------------------------
# Running scenarios for the API and the command line
# ---------------------------------------------------------------------------


def simulate(key, n=DEFAULT_N, steps=DEFAULT_STEPS, samples=40):
    """Run a scenario; return the solver and a history of diagnostics."""
    scenario = SCENARIOS[key]
    solver = scenario.make(n)
    dt = scenario.time_step(n)
    every = max(1, steps // samples)
    history = []
    for k in range(1, steps + 1):
        solver.step(dt)
        if k % every == 0 or k == steps:
            d = solver.diagnostics()
            history.append({
                "t": d["time"],
                "energy": d["kinetic_energy"],
                "enstrophy": d["enstrophy"],
                "max_divergence": d["max_divergence"],
            })
    return solver, history


def auto_scale(n, aspect):
    """Pixel size per cell so server images come out about 512 px wide."""
    return int(min(8, max(1, 512 // (n * aspect))))


@functools.lru_cache(maxsize=32)
def run_cached(key, n, steps, field, scale):
    """Simulate and render once; repeat requests with the same settings are free."""
    started = time.perf_counter()
    solver, history = simulate(key, n, steps)
    png = encode_png(render(solver, field, scale))
    payload = {
        "scenario": key,
        "title": SCENARIOS[key].title,
        "field": field,
        "n": n,
        "steps": steps,
        "seconds": round(time.perf_counter() - started, 3),
        "diagnostics": solver.diagnostics(),
        "history": history,
    }
    return png, json.dumps(payload)


def _int_arg(args, name, default, lo, hi):
    raw = args.get(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number") from None
    return max(lo, min(hi, value))


def parse_run_args(args):
    """Validate query parameters for a server run; raise ValueError if bad."""
    key = args.get("scenario", "plume")
    if key not in SCENARIOS:
        raise ValueError(f"unknown scenario {key!r}")
    scenario = SCENARIOS[key]
    field = args.get("field") or scenario.field
    if field not in FIELDS:
        raise ValueError(f"unknown field {field!r}")
    n = _int_arg(args, "n", DEFAULT_N, MIN_N, MAX_N)
    n -= n % 2
    steps = _int_arg(args, "steps", DEFAULT_STEPS, 1, MAX_STEPS)
    steps = min(steps, max(1, MAX_WORK // (n * n * scenario.aspect)))
    scale = _int_arg(args, "scale", 0, 0, 8) or auto_scale(n, scenario.aspect)
    return key, n, steps, field, scale


# ---------------------------------------------------------------------------
# Web page
# ---------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Navier–Stokes / MIK-GINGOS</title>
<style>
  /* Swiss: 12-column grid, Helvetica, flush-left type, one red accent.
     Brutalism: thick black borders, hard shadows, raw monospace, exposed structure. */
  :root {
    --ink: #000;
    --paper: #f2f0eb;
    --white: #fff;
    --red: #ff2a00;
    --line: 4px;
    --unit: 8px;
    --sans: "Helvetica Neue", Helvetica, Arial, sans-serif;
    --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--paper);
    color: var(--ink);
    font-family: var(--sans);
    line-height: 1.3;
  }
  .page {
    max-width: 1280px;
    margin: calc(var(--unit) * 3) auto;
    border: var(--line) solid var(--ink);
    background: var(--white);
  }
  .grid { display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); }
  .cell { border-right: var(--line) solid var(--ink); padding: calc(var(--unit) * 3); }
  .cell:last-child { border-right: 0; }
  .row { border-bottom: var(--line) solid var(--ink); }
  .mono {
    font-family: var(--mono);
    font-size: 0.8125rem;
    text-transform: uppercase;
    letter-spacing: 0.02em;
  }

  /* header */
  .brand {
    grid-column: span 3;
    background: var(--red);
    font-weight: 700;
    font-size: 1.25rem;
    letter-spacing: -0.02em;
  }
  .meta { grid-column: span 3; }
  nav.cell { grid-column: span 6; display: flex; padding: 0; }
  nav a {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: calc(var(--unit) * 3) var(--unit);
    border-right: var(--line) solid var(--ink);
    color: var(--ink);
    text-decoration: none;
    font-weight: 700;
    text-transform: uppercase;
  }
  nav a:last-child { border-right: 0; }
  nav a:hover { background: var(--ink); color: var(--white); }

  /* hero */
  .index {
    grid-column: span 2;
    font-family: var(--mono);
    font-size: 3rem;
    font-weight: 700;
    color: var(--red);
  }
  .hero-text { grid-column: span 10; padding-top: calc(var(--unit) * 6); }
  .hero-text h1 {
    font-size: clamp(4rem, 16vw, 13rem);
    font-weight: 700;
    line-height: 0.82;
    letter-spacing: -0.06em;
    text-transform: uppercase;
  }
  .hero-text h1 span { color: var(--red); }
  .lede {
    margin-top: calc(var(--unit) * 4);
    font-size: clamp(1.125rem, 2vw, 1.5rem);
    font-weight: 500;
    max-width: 40ch;
  }

  /* section labels */
  .label { grid-column: span 3; display: flex; flex-direction: column; gap: var(--unit); }
  .label b { font-family: var(--mono); font-size: 2rem; color: var(--red); }
  .label strong { font-size: 1.5rem; text-transform: uppercase; letter-spacing: -0.02em; }
  .hint { color: #444; text-transform: none; }

  /* controls */
  .controls { display: flex; flex-direction: column; gap: calc(var(--unit) * 2.5); margin-top: calc(var(--unit) * 2); }
  fieldset { border: 0; min-width: 0; }
  legend, .slider span {
    font-family: var(--mono);
    font-size: 0.75rem;
    text-transform: uppercase;
    margin-bottom: var(--unit);
  }
  .seg { display: grid; grid-template-columns: repeat(auto-fill, minmax(88px, 1fr)); gap: 4px; }
  .seg label { position: relative; }
  .seg input { position: absolute; opacity: 0; inset: 0; cursor: pointer; }
  .seg span {
    display: block;
    padding: 7px 6px;
    border: 3px solid var(--ink);
    text-align: center;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    cursor: pointer;
    background: var(--white);
  }
  .seg input:checked + span { background: var(--ink); color: var(--white); }
  .seg input:focus-visible + span { outline: 3px solid var(--red); outline-offset: 2px; }
  .seg label:hover span { background: var(--paper); }
  .seg label:hover input:checked + span { background: var(--ink); }

  .slider { display: grid; grid-template-columns: 1fr auto; align-items: center; column-gap: var(--unit); }
  .slider span { margin: 0; }
  .slider output { font-family: var(--mono); font-size: 0.75rem; font-weight: 700; }
  .slider input { grid-column: 1 / -1; }
  input[type=range] { -webkit-appearance: none; appearance: none; width: 100%; height: 28px; background: transparent; cursor: pointer; }
  input[type=range]::-webkit-slider-runnable-track { height: 6px; background: var(--ink); }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px; height: 26px; margin-top: -10px;
    background: var(--red); border: 3px solid var(--ink); border-radius: 0;
  }
  input[type=range]::-moz-range-track { height: 6px; background: var(--ink); }
  input[type=range]::-moz-range-thumb { width: 12px; height: 20px; background: var(--red); border: 3px solid var(--ink); border-radius: 0; }
  input[type=range]:focus-visible { outline: 3px solid var(--red); outline-offset: 2px; }

  button {
    font: inherit;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: calc(var(--unit) * 1.5) calc(var(--unit) * 3);
    border: var(--line) solid var(--ink);
    border-radius: 0;
    background: var(--red);
    color: var(--ink);
    box-shadow: 6px 6px 0 var(--ink);
    cursor: pointer;
  }
  button:hover { transform: translate(3px, 3px); box-shadow: 3px 3px 0 var(--ink); }
  button:active { transform: translate(6px, 6px); box-shadow: none; }
  button:disabled { background: var(--paper); cursor: wait; transform: none; box-shadow: none; }
  button:focus-visible { outline: 3px solid var(--ink); outline-offset: 4px; }
  .btns { display: flex; flex-wrap: wrap; gap: 12px; }
  .btns button { font-size: 0.8125rem; padding: 10px 12px; box-shadow: 4px 4px 0 var(--ink); }
  .btns button.ghost { background: var(--white); }

  /* live stage */
  .stage { grid-column: span 9; padding: 0; display: flex; flex-direction: column; }
  .canvas-wrap {
    position: relative;
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: calc(var(--unit) * 3);
    background-color: var(--white);
    background-image:
      linear-gradient(to right, #e4e1da 1px, transparent 1px),
      linear-gradient(to bottom, #e4e1da 1px, transparent 1px);
    background-size: 32px 32px;
    border-bottom: var(--line) solid var(--ink);
  }
  #sim {
    display: block;
    width: min(100%, calc(74vh * var(--aspect, 1)));
    aspect-ratio: var(--aspect, 1);
    border: var(--line) solid var(--ink);
    box-shadow: 8px 8px 0 var(--ink);
    touch-action: none;
    cursor: crosshair;
    background: var(--paper);
  }
  .badge {
    position: absolute;
    top: 16px;
    left: 16px;
    padding: 4px 10px;
    border: 3px solid var(--ink);
    background: var(--red);
    font-weight: 700;
  }
  .badge[hidden] { display: none; }
  .stats { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); }
  .stats div { padding: 12px 16px; border-right: var(--line) solid var(--ink); min-width: 0; }
  .stats div:last-child { border-right: 0; }
  .stats b {
    display: block;
    margin-top: 4px;
    font-family: var(--sans);
    font-size: 1.25rem;
    text-transform: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* equations */
  .eqs { grid-column: span 9; padding: 0; }
  .eq {
    padding: calc(var(--unit) * 4) calc(var(--unit) * 3);
    border-bottom: var(--line) solid var(--ink);
    font-size: clamp(1.25rem, 3.4vw, 2.75rem);
    letter-spacing: -0.02em;
    white-space: nowrap;
    overflow-x: auto;
  }
  .eq i { font-family: Georgia, "Times New Roman", serif; }
  .eq .vec { font-weight: 700; font-style: normal; font-family: var(--sans); }
  .eq.mass { background: var(--red); }
  .eq small { display: block; font-family: var(--mono); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.02em; margin-bottom: 8px; }
  .terms { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .term { padding: calc(var(--unit) * 3); border-right: var(--line) solid var(--ink); }
  .term:last-child { border-right: 0; }
  .term b { display: block; font-family: var(--mono); font-size: 0.8125rem; color: var(--red); }
  .term h3 { margin: 8px 0; font-size: 1.125rem; text-transform: uppercase; letter-spacing: -0.01em; }
  .term code { display: inline-block; margin-bottom: 8px; padding: 2px 6px; background: var(--ink); color: var(--white); font-family: var(--mono); font-size: 0.8125rem; }
  .term p { font-size: 0.9375rem; }

  /* server */
  .server { grid-column: span 9; padding: 0; }
  .run-form {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr)) auto;
    gap: calc(var(--unit) * 2);
    align-items: end;
    padding: calc(var(--unit) * 3);
    border-bottom: var(--line) solid var(--ink);
  }
  .run-form label { display: flex; flex-direction: column; gap: 6px; font-family: var(--mono); font-size: 0.75rem; text-transform: uppercase; min-width: 0; }
  select, input[type=number] {
    width: 100%;
    font: inherit;
    font-family: var(--sans);
    font-size: 1rem;
    font-weight: 700;
    text-transform: none;
    padding: 10px;
    border: var(--line) solid var(--ink);
    border-radius: 0;
    background: var(--paper);
    color: var(--ink);
  }
  select:focus, input[type=number]:focus { outline: none; background: var(--white); box-shadow: 4px 4px 0 var(--red); }
  .result-grid { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); }
  .frame {
    min-height: 280px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--paper);
    border-right: var(--line) solid var(--ink);
  }
  .frame img { display: block; width: 100%; height: auto; image-rendering: pixelated; }
  .frame img[hidden] { display: none; }
  .status { padding: calc(var(--unit) * 3); font-weight: 700; }
  .status.busy { background: var(--ink); color: var(--white); }
  .status.error { background: var(--red); }
  .readout { display: flex; flex-direction: column; min-width: 0; }
  table.diag { width: 100%; border-collapse: collapse; }
  .diag td { padding: 9px 16px; border-bottom: 2px solid var(--ink); }
  .diag td:last-child { text-align: right; font-weight: 700; font-family: var(--sans); text-transform: none; }
  .spark-wrap { margin-top: auto; background: var(--ink); color: var(--white); padding: 12px 16px; }
  #spark { display: block; width: 100%; height: 80px; margin-top: 8px; }
  .desc { padding: calc(var(--unit) * 3); border-top: var(--line) solid var(--ink); font-size: 1.125rem; }
  .desc p { max-width: 60ch; }

  /* method */
  .method { grid-column: span 9; padding: 0; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); list-style: none; }
  .method li { padding: calc(var(--unit) * 3); border-right: var(--line) solid var(--ink); border-bottom: var(--line) solid var(--ink); }
  .method li:nth-child(3n) { border-right: 0; }
  .method li:nth-last-child(-n+3) { border-bottom: 0; }
  .method li b { display: block; font-family: var(--mono); font-size: 2rem; color: var(--red); }
  .method h3 { margin: 8px 0; font-size: 1.125rem; text-transform: uppercase; }
  .method p { font-size: 0.9375rem; }

  footer .cell { grid-column: span 6; }
  footer .cell:last-child { background: var(--ink); color: var(--white); }

  @media (max-width: 900px) {
    .terms { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .term:nth-child(2) { border-right: 0; }
    .term:nth-child(-n+2) { border-bottom: var(--line) solid var(--ink); }
    .run-form { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .run-form button { grid-column: 1 / -1; }
    .stats { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .stats div:nth-child(3) { border-right: 0; }
    .stats div:nth-child(-n+3) { border-bottom: var(--line) solid var(--ink); }
  }
  @media (max-width: 720px) {
    .page { margin: 0; border-left: 0; border-right: 0; }
    .grid { grid-template-columns: minmax(0, 1fr); }
    .grid > * { grid-column: 1 / -1 !important; }
    .cell { border-right: 0; border-bottom: var(--line) solid var(--ink); padding: 16px; }
    .cell:last-child { border-bottom: 0; }
    .stage, .eqs, .server, .method { padding: 0; }
    .index { font-size: 2rem; }
    .hero-text { padding-top: 16px; }
    .canvas-wrap { padding: 12px 20px 20px 12px; }
    #sim { width: 100%; }
    .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .stats div { border-bottom: var(--line) solid var(--ink); }
    .stats div:nth-child(2n) { border-right: 0; }
    .stats div:nth-child(3) { border-right: var(--line) solid var(--ink); }
    .stats div:last-child { border-bottom: 0; }
    .terms, .result-grid, .method { grid-template-columns: minmax(0, 1fr); }
    .term, .method li { border-right: 0 !important; border-bottom: var(--line) solid var(--ink) !important; }
    .term:last-child, .method li:last-child { border-bottom: 0 !important; }
    .frame { border-right: 0; border-bottom: var(--line) solid var(--ink); }
    .run-form { grid-template-columns: minmax(0, 1fr); padding: 16px; }
    .eq { padding: 24px 16px; }
  }
</style>
</head>
<body>
<div class="page">
  <header class="grid row">
    <div class="cell brand">MIK&#8209;GINGOS</div>
    <div class="cell meta mono">Navier–Stokes<br>Python / Flask / NumPy</div>
    <nav class="cell" aria-label="Sections">
      <a href="#live">Live</a>
      <a href="#equations">Equations</a>
      <a href="#server">Server</a>
      <a href="#method">Method</a>
    </nav>
  </header>

  <main>
    <div class="hero grid row">
      <div class="cell index">01</div>
      <div class="cell hero-text">
        <h1>Fluid<span>.</span></h1>
        <p class="lede">The incompressible Navier–Stokes equations, solved twice: live in your browser, and in Python with NumPy on the server.</p>
      </div>
    </div>

    <section id="live" class="grid row">
      <div class="cell label">
        <b>02</b><strong>Live</strong>
        <p class="hint mono">Drag across the fluid to stir it. Space pauses.</p>
        <div class="controls">
          <fieldset>
            <legend>Scenario</legend>
            <div class="seg" id="live-scenarios">
              {% for s in scenarios %}
              <label><input type="radio" name="live-scenario" value="{{ s.key }}" {% if loop.first %}checked{% endif %}><span>{{ s.title.split(' ')[0] }}</span></label>
              {% endfor %}
            </div>
          </fieldset>
          <fieldset>
            <legend>Show</legend>
            <div class="seg" id="live-fields">
              {% for f in fields %}
              <label><input type="radio" name="live-field" value="{{ f }}" {% if loop.first %}checked{% endif %}><span>{{ f }}</span></label>
              {% endfor %}
            </div>
          </fieldset>
          <fieldset>
            <legend>Tool</legend>
            <div class="seg" id="live-tools">
              <label><input type="radio" name="live-tool" value="stir" checked><span>Stir</span></label>
              <label><input type="radio" name="live-tool" value="push"><span>Push</span></label>
              <label><input type="radio" name="live-tool" value="wall"><span>Wall</span></label>
              <label><input type="radio" name="live-tool" value="erase"><span>Erase</span></label>
            </div>
          </fieldset>
          <label class="slider"><span>Viscosity ν</span><output id="out-viscosity"></output>
            <input type="range" id="in-viscosity" min="0" max="100" value="0"></label>
          <label class="slider"><span>Vorticity ε</span><output id="out-vorticity"></output>
            <input type="range" id="in-vorticity" min="0" max="60" value="10"></label>
          <label class="slider"><span>Dye fade</span><output id="out-dissipation"></output>
            <input type="range" id="in-dissipation" min="0" max="100" value="2"></label>
          <label class="slider"><span>Pressure sweeps</span><output id="out-iters"></output>
            <input type="range" id="in-iters" min="4" max="80" value="24"></label>
          <label class="slider"><span>Brush</span><output id="out-brush"></output>
            <input type="range" id="in-brush" min="1" max="12" value="4"></label>
          <fieldset>
            <legend>Grid</legend>
            <div class="seg" id="live-grid">
              <label><input type="radio" name="live-grid" value="48"><span>48</span></label>
              <label><input type="radio" name="live-grid" value="64" checked><span>64</span></label>
              <label><input type="radio" name="live-grid" value="96"><span>96</span></label>
              <label><input type="radio" name="live-grid" value="128"><span>128</span></label>
            </div>
          </fieldset>
          <div class="btns">
            <button type="button" id="btn-pause">Pause</button>
            <button type="button" id="btn-reset" class="ghost">Reset</button>
            <button type="button" id="btn-clear" class="ghost">Clear walls</button>
          </div>
        </div>
      </div>
      <div class="cell stage">
        <div class="canvas-wrap">
          <canvas id="sim" aria-label="Fluid simulation. Drag to stir the fluid."></canvas>
          <div class="badge mono" id="badge" hidden>Paused</div>
        </div>
        <div class="stats mono" aria-live="off">
          <div>FPS<b id="st-fps">–</b></div>
          <div>Step<b id="st-step">–</b></div>
          <div>Sim time<b id="st-time">–</b></div>
          <div>Energy<b id="st-energy">–</b></div>
          <div>RMS |∇·u|<b id="st-div">–</b></div>
        </div>
      </div>
    </section>

    <section id="equations" class="grid row">
      <div class="cell label">
        <b>03</b><strong>Equations</strong>
        <p class="hint mono">Momentum and mass, for a fluid that cannot be squeezed.</p>
      </div>
      <div class="cell eqs">
        <p class="eq"><small>Momentum</small>∂<b class="vec">u</b>/∂<i>t</i> + (<b class="vec">u</b>·∇)<b class="vec">u</b> = −∇<i>p</i>/<i>ρ</i> + <i>ν</i>∇²<b class="vec">u</b> + <b class="vec">f</b></p>
        <p class="eq mass"><small>Incompressibility</small>∇·<b class="vec">u</b> = 0</p>
        <div class="terms">
          <div class="term"><b>A</b><h3>Advection</h3><code>(u·∇)u</code><p>The fluid carries its own velocity along. This is the non-linear term that makes turbulence.</p></div>
          <div class="term"><b>B</b><h3>Pressure</h3><code>−∇p/ρ</code><p>Pressure pushes back wherever fluid would pile up, so the flow stays incompressible.</p></div>
          <div class="term"><b>C</b><h3>Viscosity</h3><code>ν∇²u</code><p>Internal friction. It smooths out differences in velocity, like honey versus water.</p></div>
          <div class="term"><b>D</b><h3>Forces</h3><code>f</code><p>Everything else: buoyancy from heat, your mouse, and vorticity confinement.</p></div>
        </div>
      </div>
    </section>

    <section id="server" class="grid row">
      <div class="cell label">
        <b>04</b><strong>Server</strong>
        <p class="hint mono">Same equations, solved in Python with NumPy and rendered to PNG.</p>
      </div>
      <div class="cell server">
        <form id="run-form" class="run-form">
          <label>Scenario
            <select name="scenario" id="run-scenario">
              {% for s in scenarios %}<option value="{{ s.key }}">{{ s.title }}</option>{% endfor %}
            </select>
          </label>
          <label>Field
            <select name="field">
              <option value="">Default</option>
              {% for f in fields %}<option value="{{ f }}">{{ f|capitalize }}</option>{% endfor %}
            </select>
          </label>
          <label>Grid
            <select name="n">
              <option value="32">32</option>
              <option value="48">48</option>
              <option value="64" selected>64</option>
              <option value="96">96</option>
            </select>
          </label>
          <label>Steps
            <input type="number" name="steps" min="1" max="{{ max_steps }}" value="{{ default_steps }}">
          </label>
          <button type="submit" id="run-btn">Run →</button>
        </form>
        <div class="result-grid">
          <figure class="frame">
            <img id="frame" alt="Simulation result rendered by the Python solver" hidden>
            <div id="frame-status" class="status mono">Pick a scenario and run it.</div>
          </figure>
          <div class="readout">
            <table class="diag mono" id="diag"><tbody></tbody></table>
            <div class="spark-wrap mono">Kinetic energy over time<svg id="spark" viewBox="0 0 300 80" preserveAspectRatio="none" aria-hidden="true"></svg></div>
          </div>
        </div>
        <div class="desc"><p id="run-desc"></p></div>
      </div>
    </section>

    <section id="method" class="grid row">
      <div class="cell label">
        <b>05</b><strong>Method</strong>
        <p class="hint mono">Stable Fluids (Stam, 1999). Each step splits the equation into simple parts.</p>
      </div>
      <ol class="cell method">
        <li><b>1</b><h3>Forces</h3><p>Add buoyancy, stirring and vorticity confinement to the velocity: u += Δt·f.</p></li>
        <li><b>2</b><h3>Diffuse</h3><p>Viscosity, solved implicitly with red-black Gauss–Seidel so it never blows up.</p></li>
        <li><b>3</b><h3>Project</h3><p>Solve ∇²p = ∇·u and subtract ∇p. What is left has zero divergence.</p></li>
        <li><b>4</b><h3>Advect</h3><p>Trace each cell back along the flow and copy what was there. Stable for any time step.</p></li>
        <li><b>5</b><h3>Project</h3><p>Advection breaks incompressibility a little, so project once more.</p></li>
        <li><b>6</b><h3>Scalars</h3><p>Dye and temperature ride along with the new velocity, then spread and fade.</p></li>
      </ol>
    </section>
  </main>

  <footer class="grid">
    <div class="cell mono">&copy; MIK-GINGOS</div>
    <div class="cell mono">Built with Python, NumPy &amp; Flask</div>
  </footer>
</div>

<script id="fluid-config" type="application/json">{{ config|tojson }}</script>
<script>
{% raw %}
(() => {
"use strict";

const CONFIG = JSON.parse(document.getElementById("fluid-config").textContent);
const SCENARIOS = Object.fromEntries(CONFIG.scenarios.map((s) => [s.key, s]));

// ---------------------------------------------------------------------------
// Solver: the same Stable Fluids scheme as the Python FluidSolver.
// ---------------------------------------------------------------------------

class Fluid {
  constructor(nx, ny, width, boundary, params) {
    this.nx = nx;
    this.ny = ny;
    this.W = nx + 2;
    this.h = width / nx;
    this.width = width;
    this.height = this.h * ny;
    this.boundary = Object.assign({ mode: "box", inflow: 0, lid: 0 }, boundary);
    this.params = Object.assign({}, params);
    const size = (nx + 2) * (ny + 2);
    for (const k of ["u", "v", "u0", "v0", "p", "div", "dye", "dye0", "temp", "temp0", "curl", "tmp"]) {
      this[k] = new Float32Array(size);
    }
    this.solid = new Uint8Array(size);
    this.time = 0;
    this.emitters = [];
  }

  // Physical position of the centre of cell (i, j).
  x(i) { return (i - 0.5) * this.h; }
  y(j) { return (j - 0.5) * this.h; }

  forEachCell(fn) {
    const { nx, ny, W } = this;
    for (let j = 1; j <= ny; j++) {
      for (let i = 1; i <= nx; i++) fn(i + W * j, this.x(i), this.y(j));
    }
  }

  mask(test) {
    const ids = [];
    this.forEachCell((id, x, y) => { if (test(x, y)) ids.push(id); });
    return ids;
  }

  setBnd(kind, f) {
    const { nx, ny, W } = this;
    const mode = this.boundary.mode;
    const bottom = ny + 1, right = nx + 1;
    if (mode === "periodic") {
      for (let i = 0; i <= right; i++) {
        f[i] = f[i + W * ny];
        f[i + W * bottom] = f[i + W];
      }
      for (let j = 0; j <= bottom; j++) {
        f[W * j] = f[nx + W * j];
        f[right + W * j] = f[1 + W * j];
      }
      return;
    }
    const lid = kind === "u" && mode === "box" ? this.boundary.lid : 0;
    for (let i = 1; i <= nx; i++) {
      if (kind === "v") {
        f[i] = -f[i + W];
        f[i + W * bottom] = -f[i + W * ny];
      } else {
        f[i] = lid ? 2 * lid - f[i + W] : f[i + W];
        f[i + W * bottom] = f[i + W * ny];
      }
    }
    for (let j = 1; j <= ny; j++) {
      const r = W * j;
      if (mode === "box") {
        const s = kind === "u" ? -1 : 1;
        f[r] = s * f[r + 1];
        f[r + right] = s * f[r + nx];
      } else {
        if (kind === "u") f[r] = 2 * this.boundary.inflow - f[r + 1];
        else if (kind === "p") f[r] = f[r + 1];
        else f[r] = -f[r + 1];
        f[r + right] = kind === "p" ? -f[r + nx] : f[r + nx];
      }
    }
    f[0] = 0.5 * (f[W] + f[1]);
    f[right] = 0.5 * (f[right + W] + f[right - 1]);
    f[W * bottom] = 0.5 * (f[W * ny] + f[W * bottom + 1]);
    f[right + W * bottom] = 0.5 * (f[right + W * ny] + f[right - 1 + W * bottom]);
  }

  enforceVelocity() {
    this.setBnd("u", this.u);
    this.setBnd("v", this.v);
    const { u, v, solid } = this;
    for (let k = 0; k < solid.length; k++) if (solid[k]) { u[k] = 0; v[k] = 0; }
  }

  // Gauss-Seidel for  c * x - a * (sum of neighbours) = x0.
  linSolve(kind, x, x0, a, c, iters) {
    const { nx, ny, W } = this;
    const inv = 1 / c;
    for (let k = 0; k < iters; k++) {
      for (let j = 1; j <= ny; j++) {
        let id = 1 + W * j;
        for (let i = 1; i <= nx; i++, id++) {
          x[id] = (x0[id] + a * (x[id - 1] + x[id + 1] + x[id - W] + x[id + W])) * inv;
        }
      }
      this.setBnd(kind, x);
    }
  }

  diffuse(kind, x, rate, dt) {
    const a = rate * dt / (this.h * this.h);
    this.tmp.set(x);
    this.linSolve(kind, x, this.tmp, a, 1 + 4 * a, this.params.diffusion_iters);
  }

  advect(kind, d, d0, u, v, dt) {
    const { nx, ny, W } = this;
    const s = dt / this.h;
    const periodic = this.boundary.mode === "periodic";
    this.setBnd(kind, d0);
    for (let j = 1; j <= ny; j++) {
      let id = 1 + W * j;
      for (let i = 1; i <= nx; i++, id++) {
        let x = i - s * u[id];
        let y = j - s * v[id];
        if (periodic) {
          x = (((x - 1) % nx) + nx) % nx + 1;
          y = (((y - 1) % ny) + ny) % ny + 1;
        } else {
          x = x < 0.5 ? 0.5 : x > nx + 0.5 ? nx + 0.5 : x;
          y = y < 0.5 ? 0.5 : y > ny + 0.5 ? ny + 0.5 : y;
        }
        let i0 = Math.floor(x), j0 = Math.floor(y);
        if (i0 > nx) i0 = nx;
        if (j0 > ny) j0 = ny;
        const sx = x - i0, sy = y - j0;
        const k = i0 + W * j0;
        d[id] = (1 - sy) * ((1 - sx) * d0[k] + sx * d0[k + 1])
              + sy * ((1 - sx) * d0[k + W] + sx * d0[k + W + 1]);
      }
    }
    this.setBnd(kind, d);
  }

  project() {
    const { nx, ny, W, u, v, p, div, solid } = this;
    const h = this.h;
    const omega = this.params.sor;
    this.setBnd("u", u);
    this.setBnd("v", v);
    for (let j = 1; j <= ny; j++) {
      let id = 1 + W * j;
      for (let i = 1; i <= nx; i++, id++) {
        // div holds -h^2 * divergence, the right-hand side of the Poisson solve.
        div[id] = solid[id] ? 0 : -0.5 * h * (u[id + 1] - u[id - 1] + v[id + W] - v[id - W]);
      }
    }
    for (let k = 0; k < this.params.pressure_iters; k++) {
      for (let j = 1; j <= ny; j++) {
        let id = 1 + W * j;
        for (let i = 1; i <= nx; i++, id++) {
          if (solid[id]) continue;
          let sum = 0, count = 0;
          if (!solid[id + 1]) { sum += p[id + 1]; count++; }
          if (!solid[id - 1]) { sum += p[id - 1]; count++; }
          if (!solid[id + W]) { sum += p[id + W]; count++; }
          if (!solid[id - W]) { sum += p[id - W]; count++; }
          if (count) p[id] += omega * ((sum + div[id]) / count - p[id]);
        }
      }
      this.setBnd("p", p);
    }
    if (this.boundary.mode !== "tunnel") {
      let total = 0, cells = 0;
      this.forEachCell((id) => { if (!solid[id]) { total += p[id]; cells++; } });
      const mean = cells ? total / cells : 0;
      this.forEachCell((id) => { p[id] -= mean; });
      this.setBnd("p", p);
    }
    for (let j = 1; j <= ny; j++) {
      let id = 1 + W * j;
      for (let i = 1; i <= nx; i++, id++) {
        if (solid[id]) continue;
        const pc = p[id];
        const pE = solid[id + 1] ? pc : p[id + 1];
        const pW = solid[id - 1] ? pc : p[id - 1];
        const pS = solid[id + W] ? pc : p[id + W];
        const pN = solid[id - W] ? pc : p[id - W];
        u[id] -= (pE - pW) / (2 * h);
        v[id] -= (pS - pN) / (2 * h);
      }
    }
    this.enforceVelocity();
  }

  computeCurl() {
    const { nx, ny, W, u, v, curl } = this;
    const inv = 1 / (2 * this.h);
    this.setBnd("u", u);
    this.setBnd("v", v);
    for (let j = 1; j <= ny; j++) {
      let id = 1 + W * j;
      for (let i = 1; i <= nx; i++, id++) {
        curl[id] = (v[id + 1] - v[id - 1] - u[id + W] + u[id - W]) * inv;
      }
    }
    return curl;
  }

  confine(dt) {
    const eps = this.params.vorticity;
    if (eps <= 0) return;
    const { nx, ny, W, u, v, solid } = this;
    const curl = this.computeCurl();
    const h = this.h, inv = 1 / (2 * h);
    for (let j = 2; j < ny; j++) {
      let id = 2 + W * j;
      for (let i = 2; i < nx; i++, id++) {
        if (solid[id]) continue;
        const gx = (Math.abs(curl[id + 1]) - Math.abs(curl[id - 1])) * inv;
        const gy = (Math.abs(curl[id + W]) - Math.abs(curl[id - W])) * inv;
        const scale = dt * eps * h * curl[id] / (Math.hypot(gx, gy) + 1e-12);
        u[id] += scale * gy;
        v[id] -= scale * gx;
      }
    }
  }

  step(dt) {
    const P = this.params;
    for (const emit of this.emitters) emit(this, dt);
    if (P.buoyancy || P.weight) {
      const { v, dye, temp } = this;
      this.forEachCell((id) => { v[id] += dt * (P.weight * dye[id] - P.buoyancy * temp[id]); });
    }
    this.confine(dt);
    if (P.viscosity > 0) {
      this.diffuse("u", this.u, P.viscosity, dt);
      this.diffuse("v", this.v, P.viscosity, dt);
    }
    this.enforceVelocity();
    this.project();
    this.u0.set(this.u);
    this.v0.set(this.v);
    this.advect("u", this.u, this.u0, this.u0, this.v0, dt);
    this.advect("v", this.v, this.v0, this.u0, this.v0, dt);
    this.enforceVelocity();
    this.project();
    this.dye0.set(this.dye);
    this.temp0.set(this.temp);
    this.advect("s", this.dye, this.dye0, this.u, this.v, dt);
    this.advect("s", this.temp, this.temp0, this.u, this.v, dt);
    if (P.diffusion > 0) {
      this.diffuse("s", this.dye, P.diffusion, dt);
      this.diffuse("s", this.temp, P.diffusion, dt);
    }
    const fade = 1 / (1 + P.dissipation * dt), cool = 1 / (1 + P.cooling * dt);
    const { dye, temp, solid } = this;
    for (let k = 0; k < dye.length; k++) {
      if (solid[k]) { dye[k] = 0; temp[k] = 0; continue; }
      dye[k] *= fade;
      temp[k] *= cool;
    }
    this.time += dt;
  }

  stats() {
    const { u, v, solid, W, h } = this;
    let energy = 0, divSq = 0, cells = 0;
    this.forEachCell((id) => {
      if (solid[id]) return;
      energy += 0.5 * (u[id] * u[id] + v[id] * v[id]);
      const d = (u[id + 1] - u[id - 1] + v[id + W] - v[id - W]) / (2 * h);
      divSq += d * d;
      cells++;
    });
    return { energy: energy * h * h, rmsDiv: cells ? Math.sqrt(divSq / cells) : 0 };
  }
}

// ---------------------------------------------------------------------------
// Scenarios: settings come from Python; initial conditions are set up here.
// ---------------------------------------------------------------------------

function ellipse(f, cx, cy, rx, ry) {
  return f.mask((x, y) => ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1);
}

const SETUP = {
  plume(f) {
    const src = ellipse(f, 0.5, 0.9, 0.07, 0.025);
    f.emitters.push((s) => {
      const wobble = 0.15 * Math.sin(4 * s.time);
      for (const id of src) { s.dye[id] = 1; s.temp[id] = 1; s.u[id] = wobble; s.v[id] = -0.4; }
    });
  },
  karman(f) {
    f.forEachCell((id, x, y) => { if ((x - 0.42) ** 2 + (y - 0.5) ** 2 <= 0.075 ** 2) f.solid[id] = 1; });
    f.forEachCell((id) => { if (!f.solid[id]) f.u[id] = 1; });
    const inlet = [];
    for (let j = 1; j <= f.ny; j++) {
      if (Math.floor(f.y(j) * 12) % 2 === 0) inlet.push(1 + f.W * j, 2 + f.W * j);
    }
    const kick = ellipse(f, 0.62, 0.5, 0.05, 0.05);
    f.emitters.push((s, dt) => {
      for (const id of inlet) s.dye[id] = 1;
      if (s.time < 0.3) for (const id of kick) s.v[id] += 2 * dt;
    });
  },
  cavity(f) {
    f.forEachCell((id, x, y) => { f.dye[id] = Math.floor(y * 10) % 2 === 0 ? 1 : 0; });
  },
  shear(f) {
    const w = 0.025;
    f.forEachCell((id, x, y) => {
      const band = 0.5 * (Math.tanh((y - 0.25) / w) - Math.tanh((y - 0.75) / w));
      const bump = Math.exp(-(((y - 0.25) / 0.05) ** 2)) + Math.exp(-(((y - 0.75) / 0.05) ** 2));
      f.u[id] = 2 * band - 1;
      f.v[id] = 0.05 * Math.sin(4 * Math.PI * x) * bump;
      f.dye[id] = band;
    });
  },
  jets(f) {
    const left = ellipse(f, 0.08, 0.46, 0.04, 0.04);
    const right = ellipse(f, 0.92, 0.54, 0.04, 0.04);
    f.emitters.push((s) => {
      for (const id of left) { s.u[id] = 1.2; s.v[id] = 0; s.dye[id] = 1; }
      for (const id of right) { s.u[id] = -1.2; s.v[id] = 0; s.dye[id] = 1; s.temp[id] = 1; }
    });
  },
  taylor_green(f) {
    f.forEachCell((id, x, y) => {
      f.u[id] = Math.sin(x) * Math.cos(y);
      f.v[id] = -Math.cos(x) * Math.sin(y);
      f.dye[id] = Math.floor(x * 4 / Math.PI) % 2 === 0 ? 1 : 0;
    });
  },
};

function buildFluid(key, n) {
  const sc = SCENARIOS[key];
  const f = new Fluid(n * sc.aspect, n, sc.width, sc.boundary, sc.params);
  SETUP[key](f);
  f.enforceVelocity();
  return f;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

const LUTS = {};
for (const [name, rows] of Object.entries(CONFIG.palettes)) {
  const lut = new Uint8Array(rows.length * 3);
  rows.forEach((c, i) => { lut[3 * i] = c[0]; lut[3 * i + 1] = c[1]; lut[3 * i + 2] = c[2]; });
  LUTS[name] = lut;
}
const PAPER = [242, 240, 235];

const canvas = document.getElementById("sim");
const ctx = canvas.getContext("2d");
const off = document.createElement("canvas");
const offCtx = off.getContext("2d");
let image = null;
let values = null;
let range = 1;

function fieldValues(f, name) {
  const { nx, ny, W } = f;
  const out = values;
  const src = name === "dye" ? f.dye : name === "temperature" ? f.temp : name === "pressure" ? f.p
            : name === "vorticity" ? f.computeCurl() : null;
  let k = 0;
  for (let j = 1; j <= ny; j++) {
    let id = 1 + W * j;
    for (let i = 1; i <= nx; i++, id++, k++) {
      out[k] = src ? src[id] : Math.hypot(f.u[id], f.v[id]);
    }
  }
  return out;
}

function draw(f, name) {
  const { nx, ny, W, solid } = f;
  if (!image || image.width !== nx || image.height !== ny) {
    off.width = nx;
    off.height = ny;
    image = offCtx.createImageData(nx, ny);
    values = new Float32Array(nx * ny);
  }
  const vals = fieldValues(f, name);
  const diverging = CONFIG.diverging.includes(name);
  let lo = 0, hi = 1;
  if (name !== "dye") {
    let peak = 0;
    for (let k = 0; k < vals.length; k++) { const a = Math.abs(vals[k]); if (a > peak) peak = a; }
    // Smooth the colour range so the picture does not flicker frame to frame.
    range = Math.max(peak * 0.85, range * 0.97, 1e-6);
    if (diverging) { lo = -range; hi = range; } else { hi = range; }
  }
  const lut = LUTS[CONFIG.field_palette[name]];
  const data = image.data;
  const span = hi - lo;
  let k = 0;
  for (let j = 1; j <= ny; j++) {
    let id = 1 + W * j;
    for (let i = 1; i <= nx; i++, id++, k++) {
      const o = 4 * k;
      if (solid[id]) {
        const c = ((i + j) >> 1) % 2 === 0 ? 0 : 1;
        data[o] = c ? PAPER[0] : 0; data[o + 1] = c ? PAPER[1] : 0; data[o + 2] = c ? PAPER[2] : 0;
      } else {
        let t = (vals[k] - lo) / span;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const c = 3 * Math.round(t * 255);
        data[o] = lut[c]; data[o + 1] = lut[c + 1]; data[o + 2] = lut[c + 2];
      }
      data[o + 3] = 255;
    }
  }
  offCtx.putImageData(image, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
}

// ---------------------------------------------------------------------------
// Controls and interaction
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const radio = (name) => document.querySelector(`input[name="${name}"]:checked`).value;

const sliders = {
  viscosity: { el: $("in-viscosity"), out: $("out-viscosity"), map: (s) => (s / 100) ** 2 * 0.002, fmt: (x) => (x ? x.toExponential(1) : "0") },
  vorticity: { el: $("in-vorticity"), out: $("out-vorticity"), map: (s) => s / 10, fmt: (x) => x.toFixed(1) },
  dissipation: { el: $("in-dissipation"), out: $("out-dissipation"), map: (s) => s / 50, fmt: (x) => x.toFixed(2) + "/s" },
  pressure_iters: { el: $("in-iters"), out: $("out-iters"), map: (s) => s | 0, fmt: (x) => String(x) },
  brush: { el: $("in-brush"), out: $("out-brush"), map: (s) => s / 100, fmt: (x) => x.toFixed(2) },
};

function readSlider(key) {
  const s = sliders[key];
  const value = s.map(Number(s.el.value));
  s.out.textContent = s.fmt(value);
  return value;
}

function setSlider(key, value) {
  const s = sliders[key];
  const min = Number(s.el.min), max = Number(s.el.max);
  let best = min, bestErr = Infinity;
  for (let x = min; x <= max; x++) {
    const err = Math.abs(s.map(x) - value);
    if (err < bestErr) { bestErr = err; best = x; }
  }
  s.el.value = best;
  readSlider(key);
}

let fluid = null;
let dt = 0.01;
let paused = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let brush = 0.04;

function applySliders() {
  if (!fluid) return;
  for (const key of ["viscosity", "vorticity", "dissipation", "pressure_iters"]) {
    fluid.params[key] = readSlider(key);
  }
  brush = readSlider("brush");
}

function reset() {
  const key = radio("live-scenario");
  const n = Number(radio("live-grid"));
  const sc = SCENARIOS[key];
  fluid = buildFluid(key, n);
  dt = sc.dt * 64 / n;
  for (const key2 of ["viscosity", "vorticity", "dissipation", "pressure_iters"]) {
    setSlider(key2, sc.params[key2]);
  }
  applySliders();
  canvas.style.setProperty("--aspect", sc.aspect);
  document.querySelector(`input[name="live-field"][value="${sc.field}"]`).checked = true;
  range = 1e-6;
  resizeCanvas();
}

function setPaused(value) {
  paused = value;
  $("btn-pause").textContent = paused ? "Play" : "Pause";
  $("badge").hidden = !paused;
}

const pointer = { down: false, x: 0, y: 0, px: 0, py: 0 };

function pointerPos(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
}

canvas.addEventListener("pointerdown", (e) => {
  canvas.setPointerCapture(e.pointerId);
  [pointer.x, pointer.y] = pointerPos(e);
  pointer.px = pointer.x;
  pointer.py = pointer.y;
  pointer.down = true;
});
canvas.addEventListener("pointermove", (e) => {
  if (pointer.down) [pointer.x, pointer.y] = pointerPos(e);
});
for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
  canvas.addEventListener(type, () => { pointer.down = false; });
}
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

function applyPointer(frameDt) {
  if (!pointer.down || !fluid) return;
  const f = fluid;
  const tool = radio("live-tool");
  const cx = pointer.x * f.width, cy = pointer.y * f.height;
  let vx = (pointer.x - pointer.px) * f.width / frameDt;
  let vy = (pointer.y - pointer.py) * f.height / frameDt;
  const speed = Math.hypot(vx, vy);
  if (speed > 4) { vx *= 4 / speed; vy *= 4 / speed; }
  pointer.px = pointer.x;
  pointer.py = pointer.y;
  const r = brush * f.width;
  const r2 = r * r;
  const span = Math.ceil(r / f.h) + 1;
  const ci = Math.round(cx / f.h + 0.5), cj = Math.round(cy / f.h + 0.5);
  for (let j = Math.max(1, cj - span); j <= Math.min(f.ny, cj + span); j++) {
    for (let i = Math.max(1, ci - span); i <= Math.min(f.nx, ci + span); i++) {
      const d2 = (f.x(i) - cx) ** 2 + (f.y(j) - cy) ** 2;
      if (d2 > r2) continue;
      const id = i + f.W * j;
      if (tool === "wall") { f.solid[id] = 1; continue; }
      if (tool === "erase") { f.solid[id] = 0; continue; }
      if (f.solid[id]) continue;
      const w = Math.exp(-3 * d2 / r2);
      f.u[id] += (vx - f.u[id]) * w;
      f.v[id] += (vy - f.v[id]) * w;
      if (tool === "stir") f.dye[id] = Math.min(1, f.dye[id] + w);
    }
  }
  if (tool === "wall" || tool === "erase") f.enforceVelocity();
}

for (const el of document.querySelectorAll('input[name="live-scenario"], input[name="live-grid"]')) {
  el.addEventListener("change", reset);
}
for (const key of Object.keys(sliders)) sliders[key].el.addEventListener("input", applySliders);
$("btn-pause").addEventListener("click", () => setPaused(!paused));
$("btn-reset").addEventListener("click", reset);
$("btn-clear").addEventListener("click", () => {
  fluid.solid.fill(0);
  if (radio("live-scenario") === "karman") reset();
});
window.addEventListener("resize", resizeCanvas);
document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  if (e.code === "Space" && !["input", "select", "textarea", "button"].includes(tag)) {
    e.preventDefault();
    setPaused(!paused);
  }
});

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

let last = performance.now();
let frames = 0, fpsTime = last, stepMs = 0;

function fmt(x) {
  if (!isFinite(x)) return "–";
  if (x === 0) return "0";
  return Math.abs(x) < 0.01 || Math.abs(x) >= 1000 ? x.toExponential(2) : x.toFixed(3);
}

function frame(now) {
  const frameDt = Math.min(0.05, Math.max(0.001, (now - last) / 1000));
  last = now;
  if (!paused) {
    applyPointer(frameDt);
    const t0 = performance.now();
    fluid.step(dt);
    stepMs = stepMs * 0.9 + (performance.now() - t0) * 0.1;
  } else if (pointer.down) {
    applyPointer(frameDt);
  }
  draw(fluid, radio("live-field"));
  frames++;
  if (now - fpsTime > 500) {
    const s = fluid.stats();
    $("st-fps").textContent = Math.round(frames * 1000 / (now - fpsTime));
    $("st-step").textContent = stepMs.toFixed(1) + " ms";
    $("st-time").textContent = fluid.time.toFixed(2) + " s";
    $("st-energy").textContent = fmt(s.energy);
    $("st-div").textContent = fmt(s.rmsDiv);
    frames = 0;
    fpsTime = now;
  }
  requestAnimationFrame(frame);
}

reset();
setPaused(paused);
requestAnimationFrame(frame);

// ---------------------------------------------------------------------------
// Server runs
// ---------------------------------------------------------------------------

const form = $("run-form");
const img = $("frame");
const statusEl = $("frame-status");
const runBtn = $("run-btn");

function showDescription() {
  $("run-desc").textContent = SCENARIOS[$("run-scenario").value].description;
}

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = "status mono" + (kind ? " " + kind : "");
  statusEl.hidden = !text;
}

function fillTable(data) {
  const d = data.diagnostics;
  const rows = [
    ["Scenario", data.title],
    ["Grid", `${d.grid[0]} × ${d.grid[1]}`],
    ["Steps", String(data.steps)],
    ["Sim time", d.time.toFixed(3) + " s"],
    ["Compute", data.seconds.toFixed(2) + " s"],
    ["Kinetic energy", fmt(d.kinetic_energy)],
    ["Enstrophy", fmt(d.enstrophy)],
    ["Max speed", fmt(d.max_speed)],
    ["RMS |∇·u|", fmt(d.rms_divergence)],
    ["Dye mass", fmt(d.dye_mass)],
  ];
  const body = document.querySelector("#diag tbody");
  body.replaceChildren(...rows.map(([k, v]) => {
    const tr = document.createElement("tr");
    const a = document.createElement("td");
    const b = document.createElement("td");
    a.textContent = k;
    b.textContent = v;
    tr.append(a, b);
    return tr;
  }));
}

function spark(history) {
  const svg = $("spark");
  const ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  if (!history.length) return;
  const es = history.map((h) => h.energy);
  const ts = history.map((h) => h.t);
  const eMax = Math.max(...es) || 1;
  const t0 = ts[0], t1 = ts[ts.length - 1] || 1;
  const pts = history.map((h) => {
    const x = t1 > t0 ? (h.t - t0) / (t1 - t0) * 300 : 0;
    const y = 76 - (h.energy / eMax) * 72;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const area = document.createElementNS(ns, "polygon");
  area.setAttribute("points", `0,80 ${pts.join(" ")} 300,80`);
  area.setAttribute("fill", "#ff2a00");
  area.setAttribute("opacity", "0.35");
  const line = document.createElementNS(ns, "polyline");
  line.setAttribute("points", pts.join(" "));
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "#ff2a00");
  line.setAttribute("stroke-width", "3");
  line.setAttribute("vector-effect", "non-scaling-stroke");
  svg.append(area, line);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const params = new URLSearchParams(new FormData(form));
  if (!params.get("field")) params.delete("field");
  runBtn.disabled = true;
  img.hidden = true;
  setStatus("Computing in Python…", "busy");
  try {
    const res = await fetch("/api/run?" + params.toString());
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    img.src = data.image;
    img.hidden = false;
    setStatus("");
    fillTable(data);
    spark(data.history);
  } catch (err) {
    setStatus("Error: " + err.message, "error");
  } finally {
    runBtn.disabled = false;
  }
});
$("run-scenario").addEventListener("change", showDescription);
showDescription();
})();
{% endraw %}
</script>
</body>
</html>
"""


def page_config():
    return {
        "scenarios": scenario_config(),
        "palettes": {name: colormap(name).tolist() for name in PALETTES},
        "field_palette": FIELD_PALETTE,
        "diverging": sorted(DIVERGING),
    }


@functools.lru_cache(maxsize=1)
def _cached_page_config():
    return page_config()


@app.route("/")
def index():
    return render_template_string(
        PAGE,
        config=_cached_page_config(),
        scenarios=list(SCENARIOS.values()),
        fields=FIELDS,
        max_steps=MAX_STEPS,
        default_steps=DEFAULT_STEPS,
    )


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify(scenarios=_cached_page_config()["scenarios"], fields=list(FIELDS))


@app.route("/api/run")
def api_run():
    try:
        key, n, steps, field, scale = parse_run_args(request.args)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    png, payload = run_cached(key, n, steps, field, scale)
    data = json.loads(payload)
    data["image"] = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return jsonify(data)


@app.route("/api/frame.png")
def api_frame():
    try:
        key, n, steps, field, scale = parse_run_args(request.args)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    png, _ = run_cached(key, n, steps, field, scale)
    return Response(png, mimetype="image/png", headers={"Cache-Control": "public, max-age=3600"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _swirl(s, strength=1.0):
    """Give a solver a smooth vortex in the middle of the box."""
    cx, cy = s.width / 2, s.height / 2
    r2 = (s.X - cx) ** 2 + (s.Y - cy) ** 2
    bump = strength * np.exp(-r2 / (0.15 * s.width) ** 2)
    s.u[:] = -(s.Y - cy) * bump * 6
    s.v[:] = (s.X - cx) * bump * 6


class SolverTest(unittest.TestCase):
    def test_rejects_bad_setup(self):
        with self.assertRaises(ValueError):
            FluidSolver(2, 8)
        with self.assertRaises(ValueError):
            FluidSolver(9, 8, boundary=Boundary("periodic"))
        with self.assertRaises(ValueError):
            Boundary("donut")
        with self.assertRaises(ValueError):
            FluidParams(sor=2.5)
        with self.assertRaises(ValueError):
            FluidParams(viscosity=-1)
        with self.assertRaises(ValueError):
            FluidSolver(8, 8).step(0)

    def test_shapes_and_spacing(self):
        s = FluidSolver(32, 16, width=2.0)
        self.assertEqual(s.u.shape, (18, 34))
        self.assertAlmostEqual(s.h, 2.0 / 32)
        self.assertAlmostEqual(s.height, 1.0)
        self.assertEqual(s.field("speed").shape, (16, 32))

    def test_projection_removes_divergence(self):
        s = FluidSolver(32, 32, params=FluidParams(pressure_iters=200))
        rng = np.random.default_rng(1)
        s.u[1:-1, 1:-1] = rng.normal(size=(32, 32))
        s.v[1:-1, 1:-1] = rng.normal(size=(32, 32))
        s.u[:] = s.u + np.sin(3 * s.X)          # a smooth compressive part too
        before = np.abs(s.divergence()).mean()
        s.project()
        after = np.abs(s.divergence()).mean()
        # Grid-scale checkerboard noise is invisible to central differences on a
        # collocated grid, so noise is only partly removed; it must never grow.
        self.assertLess(after, 0.8 * before)

    def test_smooth_field_projects_to_near_zero_divergence(self):
        s = FluidSolver(48, 48, params=FluidParams(pressure_iters=400, sor=1.8))
        s.u[:] = np.sin(2 * math.pi * s.X) * np.sin(math.pi * s.Y)
        before = np.abs(s.divergence()).max()
        s.project()
        after = np.abs(s.divergence()).max()
        self.assertLess(after, 0.05 * before)

    def test_projection_keeps_divergence_free_field(self):
        s = FluidSolver(32, 32, width=2 * math.pi, boundary=Boundary("periodic"),
                        params=FluidParams(pressure_iters=50))
        s.u[:] = np.sin(s.X) * np.cos(s.Y)
        s.v[:] = -np.cos(s.X) * np.sin(s.Y)
        u0 = s.u.copy()
        s.project()
        self.assertLess(np.abs(s.u - u0).max(), 1e-3)

    def test_taylor_green_decays_at_the_exact_rate(self):
        s = SCENARIOS["taylor_green"].make(48)
        e0 = s.kinetic_energy()
        dt = 0.02
        for _ in range(50):
            s.step(dt)
        exact = taylor_green_energy(e0, s.time)
        self.assertAlmostEqual(s.time, 1.0, places=6)
        self.assertLess(abs(s.kinetic_energy() - exact) / exact, 0.05)

    def test_uniform_scalar_stays_uniform(self):
        s = FluidSolver(24, 24, boundary=Boundary("periodic"))
        _swirl(s)
        s.dye[:] = 0.7
        s.step(0.01)
        self.assertTrue(np.allclose(s.dye[1:-1, 1:-1], 0.7))

    def test_bilinear_sampling_is_exact_for_linear_fields(self):
        s = FluidSolver(16, 16)
        f = 2.0 * s.X + 3.0 * s.Y
        x = np.array([2.25, 7.5, 10.9])
        y = np.array([3.75, 1.5, 12.1])
        expected = 2.0 * (x - 0.5) * s.h + 3.0 * (y - 0.5) * s.h
        self.assertTrue(np.allclose(s._sample(f, x, y), expected))

    def test_maccormack_creates_no_new_extremes(self):
        s = FluidSolver(32, 32, params=FluidParams(advection="maccormack"))
        _swirl(s)
        s.project()
        s.dye[1:-1, 1:-1] = (s.X[1:-1, 1:-1] < 0.5).astype(float)
        for _ in range(10):
            s.step(0.01)
        self.assertGreaterEqual(s.dye.min(), -1e-12)
        self.assertLessEqual(s.dye.max(), 1.0 + 1e-12)

    def test_dye_is_roughly_conserved_in_a_closed_box(self):
        s = FluidSolver(48, 48)
        _swirl(s)
        s.project()
        s.dye[:] = np.exp(-((s.X - 0.4) ** 2 + (s.Y - 0.5) ** 2) / 0.01)
        m0 = s.dye_mass()
        for _ in range(20):
            s.step(0.005)
        self.assertLess(abs(s.dye_mass() - m0) / m0, 0.05)

    def test_solid_body_rotation_has_vorticity_two_omega(self):
        s = FluidSolver(16, 16, boundary=Boundary("periodic"))
        omega = 1.5
        s.u[:] = -omega * (s.Y - 0.5)
        s.v[:] = omega * (s.X - 0.5)
        w = s.vorticity()[2:-2, 2:-2]
        self.assertTrue(np.allclose(w, 2 * omega))

    def test_obstacles_stay_still(self):
        s = FluidSolver(48, 24, 2.0, Boundary("tunnel", inflow=1.0))
        s.add_circle(0.5, 0.5, 0.1)
        s.u[1:-1, 1:-1] = 1.0
        for _ in range(5):
            s.step(0.01)
        self.assertTrue(s.solid.any())
        self.assertEqual(np.abs(s.u[s.solid]).max(), 0.0)
        self.assertEqual(np.abs(s.v[s.solid]).max(), 0.0)
        self.assertEqual(np.abs(s.dye[s.solid]).max(), 0.0)

    def test_tunnel_inflow_sets_wall_speed(self):
        s = FluidSolver(16, 8, 2.0, Boundary("tunnel", inflow=1.5))
        s.set_bnd(s.u, "u")
        face = 0.5 * (s.u[1:-1, 0] + s.u[1:-1, 1])
        self.assertTrue(np.allclose(face, 1.5))

    def test_lid_drives_the_cavity(self):
        s = SCENARIOS["cavity"].make(24)
        for _ in range(20):
            s.step(SCENARIOS["cavity"].time_step(24))
        top = s.u[1, 1:-1].mean()
        bottom = s.u[-2, 1:-1].mean()
        self.assertGreater(top, 0.1)
        self.assertLess(bottom, top)

    def test_periodic_ghosts_wrap(self):
        s = FluidSolver(8, 8, boundary=Boundary("periodic"))
        s.dye[1:-1, 1:-1] = np.arange(64).reshape(8, 8)
        s.set_bnd(s.dye, "s")
        self.assertTrue(np.array_equal(s.dye[0, 1:-1], s.dye[8, 1:-1]))
        self.assertTrue(np.array_equal(s.dye[1:-1, -1], s.dye[1:-1, 1]))

    def test_advance_respects_cfl(self):
        s = FluidSolver(16, 16, boundary=Boundary("periodic"))
        s.u[:] = 2.0
        steps = s.advance(0.2, cfl=0.5)
        self.assertAlmostEqual(s.time, 0.2)
        self.assertGreaterEqual(steps, math.ceil(0.2 / (0.5 * s.h / 2.0)))

    def test_viscosity_slows_the_flow(self):
        still = FluidSolver(32, 32)
        sticky = FluidSolver(32, 32, params=FluidParams(viscosity=0.01))
        for s in (still, sticky):
            _swirl(s)
            s.project()
            for _ in range(10):
                s.step(0.01)
        self.assertLess(sticky.kinetic_energy(), still.kinetic_energy())

    def test_buoyancy_lifts_hot_fluid(self):
        s = FluidSolver(24, 24, params=FluidParams(buoyancy=5.0))
        s.temp[:] = _ellipse(s, 0.5, 0.7, 0.1, 0.1).astype(float)
        s.step(0.02)
        hot = _ellipse(s, 0.5, 0.7, 0.05, 0.05)
        self.assertLess(s.v[hot].mean(), 0.0)    # negative v means upward

    def test_every_scenario_runs(self):
        for key, scenario in SCENARIOS.items():
            with self.subTest(scenario=key):
                s = scenario.make(16)
                for _ in range(3):
                    s.step(scenario.time_step(16))
                for name in FIELDS:
                    self.assertTrue(np.isfinite(s.field(name)).all())
                self.assertEqual(s.ny, 16)
                self.assertEqual(s.nx, 16 * scenario.aspect)

    def test_unknown_field(self):
        with self.assertRaises(ValueError):
            FluidSolver(8, 8).field("colour")


class RenderTest(unittest.TestCase):
    def test_colormap_endpoints(self):
        lut = colormap("ink")
        self.assertEqual(lut.shape, (256, 3))
        self.assertEqual(lut[0].tolist(), _hex_rgb(PAPER))
        self.assertEqual(lut[-1].tolist(), [0, 0, 0])

    def test_diverging_fields_centre_on_zero(self):
        t = normalize("vorticity", np.array([-2.0, 0.0, 2.0]))
        self.assertAlmostEqual(t[1], 0.5)
        self.assertLess(t[0], 0.5)
        self.assertGreater(t[2], 0.5)

    def test_render_shape_and_hatching(self):
        s = FluidSolver(20, 10, 2.0, Boundary("tunnel", inflow=1.0))
        s.add_circle(1.0, 0.5, 0.2)
        rgb = render(s, "speed", scale=3)
        self.assertEqual(rgb.shape, (30, 60, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        solid_px = np.repeat(np.repeat(s.solid[1:-1, 1:-1], 3, 0), 3, 1)
        colours = {tuple(c) for c in rgb[solid_px]}
        self.assertEqual(colours, {(0, 0, 0), tuple(_hex_rgb(PAPER))})

    def test_png_round_trip(self):
        rgb = np.random.default_rng(2).integers(0, 256, size=(7, 11, 3), dtype=np.uint8)
        data = encode_png(rgb)
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(np.array_equal(decode_png(data), rgb))

    def test_png_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            encode_png(np.zeros((4, 4)))
        with self.assertRaises(ValueError):
            decode_png(b"not a png")


class AppTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'<canvas id="sim"', res.data)
        self.assertIn(b"Navier", res.data)
        self.assertIn(b"fluid-config", res.data)
        self.assertNotIn(b"{%", res.data)

    def test_scenarios_endpoint(self):
        data = self.client.get("/api/scenarios").get_json()
        keys = [s["key"] for s in data["scenarios"]]
        self.assertEqual(keys, list(SCENARIOS))
        self.assertIn("params", data["scenarios"][0])

    def test_run_endpoint(self):
        res = self.client.get("/api/run?scenario=jets&n=16&steps=3&field=speed")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["image"].startswith("data:image/png;base64,"))
        self.assertEqual(data["steps"], 3)
        self.assertEqual(data["diagnostics"]["grid"], [16, 16])
        png = base64.b64decode(data["image"].split(",", 1)[1])
        self.assertEqual(decode_png(png).shape[2], 3)

    def test_frame_endpoint(self):
        res = self.client.get("/api/frame.png?scenario=karman&n=16&steps=2")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "image/png")
        scale = auto_scale(16, 2)
        self.assertEqual(decode_png(res.data).shape, (16 * scale, 32 * scale, 3))

    def test_bad_requests(self):
        for query in ("scenario=nope", "field=colour", "steps=lots", "n=abc"):
            with self.subTest(query=query):
                res = self.client.get("/api/run?" + query)
                self.assertEqual(res.status_code, 400)
                self.assertIn("error", res.get_json())

    def test_arguments_are_clamped(self):
        key, n, steps, field, scale = parse_run_args({"n": "9999", "steps": "-5", "scenario": "plume"})
        self.assertEqual((key, n, steps, field), ("plume", MAX_N, 1, "dye"))
        _, n, _, _, _ = parse_run_args({"n": "33"})
        self.assertEqual(n, 32)
        _, n, steps, _, _ = parse_run_args({"scenario": "karman", "n": "128", "steps": "600"})
        self.assertLessEqual(n * n * 2 * steps, MAX_WORK)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _cmd_render(args):
    scenario = SCENARIOS[args.scenario]
    field = args.field or scenario.field
    started = time.perf_counter()
    solver, _ = simulate(args.scenario, args.n, args.steps)
    scale = args.scale or auto_scale(args.n, scenario.aspect)
    with open(args.out, "wb") as fh:
        fh.write(encode_png(render(solver, field, scale)))
    d = solver.diagnostics()
    print(f"{scenario.title}: {args.steps} steps on {d['grid'][0]}x{d['grid'][1]} "
          f"in {time.perf_counter() - started:.2f}s -> {args.out}")
    for k in ("time", "kinetic_energy", "enstrophy", "max_speed", "rms_divergence", "dye_mass"):
        print(f"  {k:<15} {d[k]:.6g}")


def _cmd_scenarios(_args):
    for s in SCENARIOS.values():
        print(f"{s.key:<13} {s.title}")
        print(f"{'':<13} {s.description}")


def _cmd_bench(args):
    print(f"{'scenario':<13} {'grid':>9} {'ms/step':>9}")
    for key, scenario in SCENARIOS.items():
        s = scenario.make(args.n)
        dt = scenario.time_step(args.n)
        s.step(dt)
        started = time.perf_counter()
        for _ in range(args.steps):
            s.step(dt)
        ms = (time.perf_counter() - started) * 1000 / args.steps
        print(f"{key:<13} {s.nx:>4}x{s.ny:<4} {ms:>9.2f}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--test" in argv:
        unittest.main(argv=[sys.argv[0], "-v"])
        return

    parser = argparse.ArgumentParser(description="Navier-Stokes fluid simulator.")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the web app (default)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5000)
    serve.add_argument("--debug", action="store_true")

    rend = sub.add_parser("render", help="simulate a scenario and save a PNG")
    rend.add_argument("--scenario", choices=list(SCENARIOS), default="plume")
    rend.add_argument("--field", choices=list(FIELDS))
    rend.add_argument("--n", type=int, default=DEFAULT_N, help="grid rows")
    rend.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    rend.add_argument("--scale", type=int, default=0, help="pixels per cell (0 = auto)")
    rend.add_argument("--out", default="fluid.png")

    sub.add_parser("scenarios", help="list the built-in scenarios")

    bench = sub.add_parser("bench", help="time one step of every scenario")
    bench.add_argument("--n", type=int, default=DEFAULT_N)
    bench.add_argument("--steps", type=int, default=20)

    args = parser.parse_args(argv)
    if args.command == "render":
        if args.n < MIN_N or args.steps < 1:
            parser.error(f"--n must be at least {MIN_N} and --steps at least 1")
        _cmd_render(args)
    elif args.command == "scenarios":
        _cmd_scenarios(args)
    elif args.command == "bench":
        _cmd_bench(args)
    else:
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 5000)
        app.run(host=host, port=port, debug=getattr(args, "debug", False))


if __name__ == "__main__":
    main()
