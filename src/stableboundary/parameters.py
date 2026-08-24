"""Immutable parameter contracts for stable laws near the Gaussian boundary.

All conventional stable parameters use Nolan's continuous ``S0``
parameterization.  In this convention ``alpha == 2`` is a Gaussian law with
variance ``2 * scale**2``; ``scale`` is therefore not its standard deviation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import ClassVar, Literal

from ._exceptions import UnidentifiedParameterError, ValidationError


def _finite_float(name: str, value: float) -> float:
    """Return a canonical finite float or raise a package validation error."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValidationError(f"{name} must be finite")
    return converted


def _positive_float(name: str, value: float) -> float:
    """Return a canonical finite positive float."""
    converted = _finite_float(name, value)
    if converted <= 0.0:
        raise ValidationError(f"{name} must be strictly positive")
    return converted


@dataclass(frozen=True, slots=True)
class StableParams:
    """Conventional univariate stable-law parameters in Nolan's ``S0`` form.

    At ``alpha == 2`` the distribution is ``N(loc, 2 * scale**2)`` and does
    not depend on ``beta``.  The object may represent that distribution, but
    conversion to boundary allocation coordinates is refused because neither
    ``beta`` nor ``p`` is identified there.
    """

    alpha: float
    beta: float
    loc: float = 0.0
    scale: float = 1.0

    parameterization: ClassVar[Literal["S0"]] = "S0"

    def __post_init__(self) -> None:
        alpha = _finite_float("alpha", self.alpha)
        beta = _finite_float("beta", self.beta)
        loc = _finite_float("loc", self.loc)
        scale = _positive_float("scale", self.scale)
        if not 0.0 < alpha <= 2.0:
            raise ValidationError("alpha must lie in (0, 2]")
        if not -1.0 <= beta <= 1.0:
            raise ValidationError("beta must lie in [-1, 1]")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "loc", loc)
        object.__setattr__(self, "scale", scale)

    def to_local(self, *, r: float) -> LocalCoordinates:
        """Convert to ``(r, h, p)`` while retaining the supplied design scale."""
        r_value = _positive_float("r", r)
        if self.alpha == 2.0:
            raise UnidentifiedParameterError(
                "beta and p are not identified at the exact Gaussian point alpha=2"
            )
        return LocalCoordinates(
            r=r_value,
            h=(2.0 - self.alpha) / r_value,
            p=(1.0 + self.beta) / 2.0,
        )

    def to_signed_tail_gap(self, *, r: float) -> SignedTailGap:
        """Convert to signed empirical-scale gaps while retaining ``r``."""
        return self.to_local(r=r).to_signed_tail_gap()

    @classmethod
    def from_local(
        cls,
        coordinates: LocalCoordinates,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> StableParams:
        """Construct conventional parameters from local coordinates."""
        return coordinates.to_stable(loc=loc, scale=scale)

    @classmethod
    def from_signed_tail_gap(
        cls,
        gap: SignedTailGap,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> StableParams:
        """Construct conventional parameters from signed tail gaps."""
        return gap.to_stable(loc=loc, scale=scale)


@dataclass(frozen=True, slots=True)
class LocalCoordinates:
    """Interior local coordinates with ``alpha = 2-r*h`` and ``beta = 2*p-1``."""

    r: float
    h: float
    p: float

    def __post_init__(self) -> None:
        r = _positive_float("r", self.r)
        h = _positive_float("h", self.h)
        p = _finite_float("p", self.p)
        if not 0.0 < p < 1.0:
            raise ValidationError("p must lie strictly inside (0, 1)")
        alpha = 2.0 - r * h
        if not isfinite(alpha) or not 0.0 < alpha < 2.0:
            raise ValidationError("r*h must map alpha strictly inside (0, 2)")
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "p", p)

    @property
    def alpha(self) -> float:
        """Conventional characteristic exponent ``2-r*h``."""
        return 2.0 - self.r * self.h

    @property
    def beta(self) -> float:
        """Conventional skewness parameter ``2*p-1``."""
        return 2.0 * self.p - 1.0

    @property
    def tau_plus(self) -> float:
        """Positive signed tail gap ``r*h*p``."""
        return self.r * self.h * self.p

    @property
    def tau_minus(self) -> float:
        """Negative signed tail gap ``r*h*(1-p)``."""
        return self.r * self.h * (1.0 - self.p)

    def to_stable(self, *, loc: float = 0.0, scale: float = 1.0) -> StableParams:
        """Convert to conventional ``S0`` parameters."""
        return StableParams(
            alpha=self.alpha,
            beta=self.beta,
            loc=loc,
            scale=scale,
        )

    def to_signed_tail_gap(self) -> SignedTailGap:
        """Convert to signed empirical-scale gaps."""
        return SignedTailGap(
            r=self.r,
            tau_plus=self.tau_plus,
            tau_minus=self.tau_minus,
        )

    @classmethod
    def from_stable(cls, params: StableParams, *, r: float) -> LocalCoordinates:
        """Construct local coordinates from conventional ``S0`` parameters."""
        return params.to_local(r=r)

    @classmethod
    def from_signed_tail_gap(cls, gap: SignedTailGap) -> LocalCoordinates:
        """Construct local coordinates from signed empirical-scale gaps."""
        return gap.to_local()


@dataclass(frozen=True, slots=True)
class SignedTailGap:
    """Positive and negative empirical-scale gaps with their design scale.

    The identities are ``tau_plus = r*h*p`` and
    ``tau_minus = r*h*(1-p)``.  Both components are strictly positive because
    the current theorem covers only compact interior allocations.
    """

    r: float
    tau_plus: float
    tau_minus: float

    def __post_init__(self) -> None:
        r = _positive_float("r", self.r)
        tau_plus = _positive_float("tau_plus", self.tau_plus)
        tau_minus = _positive_float("tau_minus", self.tau_minus)
        total = tau_plus + tau_minus
        alpha = 2.0 - total
        h = total / r
        if not isfinite(total) or not isfinite(h) or not 0.0 < alpha < 2.0:
            raise ValidationError(
                "signed tail gaps must map alpha strictly inside (0, 2)"
            )
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "tau_plus", tau_plus)
        object.__setattr__(self, "tau_minus", tau_minus)

    @property
    def h(self) -> float:
        """Local total tail intensity."""
        return (self.tau_plus + self.tau_minus) / self.r

    @property
    def p(self) -> float:
        """Positive-tail allocation proportion."""
        return self.tau_plus / (self.tau_plus + self.tau_minus)

    @property
    def alpha(self) -> float:
        """Conventional characteristic exponent."""
        return 2.0 - self.tau_plus - self.tau_minus

    @property
    def beta(self) -> float:
        """Conventional skewness parameter."""
        return 2.0 * self.p - 1.0

    def to_local(self) -> LocalCoordinates:
        """Convert to local coordinates while retaining ``r``."""
        return LocalCoordinates(r=self.r, h=self.h, p=self.p)

    def to_stable(self, *, loc: float = 0.0, scale: float = 1.0) -> StableParams:
        """Convert to conventional ``S0`` parameters."""
        return self.to_local().to_stable(loc=loc, scale=scale)

    @classmethod
    def from_local(cls, coordinates: LocalCoordinates) -> SignedTailGap:
        """Construct signed gaps from local coordinates."""
        return coordinates.to_signed_tail_gap()

    @classmethod
    def from_stable(cls, params: StableParams, *, r: float) -> SignedTailGap:
        """Construct signed gaps from conventional ``S0`` parameters."""
        return params.to_signed_tail_gap(r=r)


__all__ = ["LocalCoordinates", "SignedTailGap", "StableParams"]
