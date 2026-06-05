# ---------------------------------------------------------------------------
# Domain-randomization transforms
#
# Two families that act on different things:
#   * ValueTransform   — change pixel *values* (tone curves); applied to the
#                        extracted pixel array.
#   * SpatialTransform — change *which geography maps to which pixel* (geometry);
#                        applied to the (array, Affine) pair of the working window.
#
# Each transform carries a per-episode application probability ``p`` (default 1.0):
# each episode it is applied with probability ``p``, otherwise it is the identity
# for that episode. This lets a fraction of episodes stay fully clean to protect
# in-distribution performance (standard image-augmentation practice). Parameters are
# ``[low, high]`` ranges sampled uniformly per episode (collapsed range = constant).
# ---------------------------------------------------------------------------
from typing import Callable, Literal, Optional, Annotated, List

import numpy as np
from affine import Affine
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ValueTransform(BaseModel):
    """Per-pixel value transform (tone curve). ``sample`` returns the concrete
    function for this episode (identity when the probability roll fails)."""
    model_config = ConfigDict(extra="forbid")
    p: float = Field(default=1.0, ge=0.0, le=1.0)

    def sample(self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        if rng.random() >= self.p:
            return lambda values: values
        return self._sample(rng)

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        raise NotImplementedError("_sample() must be implemented by ValueTransform subclasses")


class GammaCorrection(ValueTransform):
    """Power law: compress the range of values (lift countryside toward cities)."""
    type: Literal["gamma_correction"] = "gamma_correction"
    gamma: tuple[float, float]
    identity_intersect: float

    def calculate_offset(self, gamma: float) -> float:
        return self.identity_intersect / (self.identity_intersect ** gamma)

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        gamma = rng.uniform(*self.gamma)
        offset =self.calculate_offset(gamma)
        print(f"gamma: {gamma}, offset: {offset}")
        return lambda values: np.power(values, gamma) * offset


class FloorRaise(ValueTransform):
    type: Literal["floor_raise"] = "floor_raise"
    floor: tuple[float, float]

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        floor = rng.uniform(*self.floor)
        return lambda values: values + floor


class ScaleValues(ValueTransform):
    type: Literal["scale_values"] = "scale_values"
    factor: tuple[float, float]

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        factor = rng.uniform(*self.factor)
        return lambda values: values * factor


class Clip(ValueTransform):
    """Clip values to ``[0, upper]`` (people/km²). Applied last; this is the
    cross-resolution consistency mechanism, so keep ``p=1.0``.

    Provide exactly one of:
      * ``upper``      — an explicit ``[low, high]`` range (people/km²), sampled per episode.
      * ``percentile`` — clip at this percentile of the *base* map (e.g. 99.9), matching
                         the legacy ``map_source_max`` clipping. Resolved once by the source
                         (constant, since the base raster never changes)."""
    type: Literal["clip"] = "clip"
    upper: Optional[tuple[float, float]] = None
    percentile: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def _exactly_one(self) -> "Clip":
        if (self.upper is None) == (self.percentile is None):
            raise ValueError("Clip requires exactly one of 'upper' or 'percentile'.")
        return self

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        assert self.upper is not None, (
            "Clip.percentile must be resolved to 'upper' before sampling "
            "(handled by TransformedTiffMapSource)."
        )
        upper = rng.uniform(*self.upper)
        return lambda values: np.clip(values, 0.0, upper)


ValueTransformType = Annotated[GammaCorrection | FloorRaise | ScaleValues | Clip, Field(discriminator="type")]


class SpatialTransform(BaseModel):
    """Geometry transform on the working window. ``sample`` returns a function
    ``(array, transform) -> (array, transform)`` (identity when the roll fails)."""
    model_config = ConfigDict(extra="forbid")
    p: float = Field(default=1.0, ge=0.0, le=1.0)

    def sample(self, rng: np.random.Generator) -> Callable[[np.ndarray, Affine], tuple[np.ndarray, Affine]]:
        if rng.random() >= self.p:
            return lambda array, transform: (array, transform)
        return self._sample(rng)

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray, Affine], tuple[np.ndarray, Affine]]:
        raise NotImplementedError("_sample() must be implemented by SpatialTransform subclasses")


class Zoom(SpatialTransform):
    """Scale the field about the window centre by factor ``z``: z>1 magnifies
    (bigger cities), z<1 compresses. Rescales the window's Affine pixel size about
    its centre; ``regenerate`` reads enough base data to cover the magnified footprint."""
    type: Literal["zoom"] = "zoom"
    factor: tuple[float, float]

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray, Affine], tuple[np.ndarray, Affine]]:
        z = rng.uniform(*self.factor)

        def apply(array: np.ndarray, transform: Affine) -> tuple[np.ndarray, Affine]:
            rows, cols = array.shape
            cx, cy = transform * (cols / 2.0, rows / 2.0)
            scaled = (
                Affine.translation(cx, cy)
                * Affine.scale(1.0 / z)
                * Affine.translation(-cx, -cy)
                * transform
            )
            return array, scaled

        return apply

    @property
    def max_factor(self) -> float:
        return self.factor[1]


class Flip(SpatialTransform):
    """Mirror the window array, keeping the geographic extent (Affine) unchanged —
    a reflection of the field about the window centre."""
    type: Literal["flip"] = "flip"
    axis: Literal["ns", "ew", "both", "random"] = "random"

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray, Affine], tuple[np.ndarray, Affine]]:
        axis = self.axis if self.axis != "random" else rng.choice(["ns", "ew", "both"])

        def apply(array: np.ndarray, transform: Affine) -> tuple[np.ndarray, Affine]:
            if axis in ("ns", "both"):
                array = np.flipud(array)
            if axis in ("ew", "both"):
                array = np.fliplr(array)
            return np.ascontiguousarray(array), transform

        return apply


SpatialTransformType = Annotated[Zoom | Flip, Field(discriminator="type")]


class ValuePipeline:
    """Holds value transforms; samples concrete functions once per episode."""
    def __init__(self, transforms: List[ValueTransformType]):
        self._transforms = transforms
        self._fns: List[Callable[[np.ndarray], np.ndarray]] = []

    def sample(self, rng: np.random.Generator) -> None:
        self._fns = [t.sample(rng) for t in self._transforms]

    def apply(self, array: np.ndarray) -> np.ndarray:
        for fn in self._fns:
            array = fn(array)
        return array


class SpatialPipeline:
    """Holds spatial transforms; samples concrete functions once per episode."""
    def __init__(self, transforms: List[SpatialTransformType]):
        self._transforms = transforms
        self._fns: List[Callable[[np.ndarray, Affine], tuple[np.ndarray, Affine]]] = []

    def sample(self, rng: np.random.Generator) -> None:
        self._fns = [t.sample(rng) for t in self._transforms]

    def apply(self, array: np.ndarray, transform: Affine) -> tuple[np.ndarray, Affine]:
        for fn in self._fns:
            array, transform = fn(array, transform)
        return array, transform

    @property
    def max_read_expansion(self) -> float:
        """How much larger than the env bounds the base read must be so that
        zoom-in (magnify) still covers the window. 1.0 when there are no zooms."""
        zooms = [t.max_factor for t in self._transforms if isinstance(t, Zoom)]
        return max([1.0, *zooms])