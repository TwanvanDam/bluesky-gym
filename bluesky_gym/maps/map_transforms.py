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
from scipy.stats import loguniform
from affine import Affine
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


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
    """Power law: compress the range of values (lift countryside toward cities).

    Uses the median-matched multiplier ``c = target_value / base_median^γ`` so the
    transformed field's median equals ``target_value``
    ``percentile`` selects the base-map statistic that represents the
    "base median"; call ``resolve(value)`` with the precomputed people/km² value
    before sampling (handled by TransformedTiffMapSource)."""
    type: Literal["gamma_correction"] = "gamma_correction"
    gamma: tuple[float, float]
    percentile: float = Field(ge=0.0, le=100.0)
    target_value: float = Field(gt=0.0)
    _base_value: Optional[float] = PrivateAttr(default=None)

    def resolve(self, value: float) -> None:
        self._base_value = value

    def _calculate_c(self, gamma: float) -> float:
        assert self._base_value is not None
        return self.target_value / (self._base_value ** gamma)

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        assert self._base_value is not None, (
            "GammaCorrection.percentile must be resolved before sampling — call resolve(value) first "
            "(handled by TransformedTiffMapSource)."
        )
        gamma = float(rng.uniform(*self.gamma))
        c = self._calculate_c(gamma)
        return lambda values: np.power(values, gamma) * c


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

    ``percentile`` — clip at this percentile of the *base* map (e.g. 99.9), matching
    the legacy ``map_source_max`` clipping. Call ``resolve(base_array)`` once after
    loading the base raster (handled by TransformedTiffMapSource); the resolved upper
    bound is constant across episodes since the base raster never changes."""
    type: Literal["clip"] = "clip"
    percentile: float = Field(ge=0.0, le=100.0)
    _upper_limit: Optional[float] = PrivateAttr(default=None)

    def resolve(self, upper_value: float) -> None:
        self._upper_limit = upper_value

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
        assert self._upper_limit is not None, (
            "Clip.percentile must be resolved before sampling — call resolve(base_array) first "
            "(handled by TransformedTiffMapSource)."
        )
        upper = self._upper_limit
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
        z = loguniform.rvs(self.factor[0], self.factor[1], size=1, random_state=rng)[0]

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
    axis: Literal["ns", "ew", "combination"] = "combination"

    def _sample(self, rng: np.random.Generator) -> Callable[[np.ndarray, Affine], tuple[np.ndarray, Affine]]:
        if self.axis in ("ns", "ew"):
            flips = [self.axis]
        elif self.axis == "combination":
            # NS and EW are equivalent under rotation, so one random single-axis flip is enough.
            # "both" = 180° rotation, already in the training distribution — excluded.
            flips = [str(rng.choice(["ns", "ew"]))]

        def apply(array: np.ndarray, transform: Affine) -> tuple[np.ndarray, Affine]:
            for flip in flips:
                if flip == "ns":
                    array = np.flipud(array)
                if flip == "ew":
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

if __name__ == '__main__':
    base_array = np.linspace(1, 16, 16).reshape((4,4))
    print(base_array)
    rng = np.random.default_rng(4)
    pipeline = SpatialPipeline([Flip(p=1, type="flip", axis="combination")])
    pipeline.sample(rng)
    result = pipeline.apply(base_array, transform=Affine(0.5, 0.5, 0,0,0,0))
    print(result)