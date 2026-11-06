"""Forced noisy ring of phases: one reference domain, not the ENTRELACS core.

The model is an Euler-Maruyama-style update:
    dphi_i = [omega_i + A sin(omega_drive*t - phi_i) + J*K_i(phi)] dt
             + sigma dW_i
where K is nearest-neighbour ring coupling. The implementation is a fresh,
small baseline informed by RATISS-SIMULTANEITE; it does not copy that file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ring_coupling(phases: ArrayLike) -> NDArray[np.float64]:
    """Nearest-neighbour sine coupling on a periodic one-dimensional ring."""
    values = np.asarray(phases, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("phases must be a 1-D array with at least 3 units")
    if not np.all(np.isfinite(values)):
        raise ValueError("phases must be finite")
    return np.sin(np.roll(values, 1) - values) + np.sin(np.roll(values, -1) - values)


@dataclass(frozen=True)
class ForcedRing:
    """Reference phase network with shared drive, local noise and ring coupling."""

    intrinsic_frequencies: ArrayLike
    coupling: float = 0.0
    drive_amplitude: float = 0.0
    drive_frequency: float = 1.0
    noise_sigma: float = 0.0

    def __post_init__(self) -> None:
        omega = np.asarray(self.intrinsic_frequencies, dtype=np.float64).copy()
        if omega.ndim != 1 or omega.size < 3 or not np.all(np.isfinite(omega)):
            raise ValueError("intrinsic_frequencies must be finite, 1-D and length >= 3")
        for name in ("coupling", "drive_amplitude", "drive_frequency", "noise_sigma"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "noise_sigma" and value < 0.0:
                raise ValueError("noise_sigma must be non-negative")
            object.__setattr__(self, name, value)
        omega.setflags(write=False)
        object.__setattr__(self, "intrinsic_frequencies", omega)

    @classmethod
    def homogeneous(
        cls,
        n_units: int,
        *,
        frequency: float = 1.0,
        coupling: float = 0.0,
        drive_amplitude: float = 0.0,
        drive_frequency: float = 1.0,
        noise_sigma: float = 0.0,
    ) -> "ForcedRing":
        """Convenience constructor for identical intrinsic frequencies."""
        if isinstance(n_units, bool) or not isinstance(n_units, (int, np.integer)) or n_units < 3:
            raise ValueError("n_units must be an integer >= 3")
        if not np.isfinite(frequency):
            raise ValueError("frequency must be finite")
        return cls(
            np.full(int(n_units), float(frequency), dtype=np.float64),
            coupling=coupling,
            drive_amplitude=drive_amplitude,
            drive_frequency=drive_frequency,
            noise_sigma=noise_sigma,
        )

    def step(
        self,
        time: float,
        state: NDArray[np.float64],
        dt: float,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        phases = np.asarray(state, dtype=np.float64)
        if phases.shape != self.intrinsic_frequencies.shape:
            raise ValueError("state shape must match intrinsic_frequencies")
        if not np.all(np.isfinite(phases)) or not np.isfinite(time):
            raise ValueError("time and phases must be finite")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")

        interaction = self.coupling * ring_coupling(phases)
        drive = self.drive_amplitude * np.sin(self.drive_frequency * time - phases)
        noise = self.noise_sigma * np.sqrt(dt) * rng.standard_normal(phases.shape)
        return phases + (self.intrinsic_frequencies + drive + interaction) * dt + noise
