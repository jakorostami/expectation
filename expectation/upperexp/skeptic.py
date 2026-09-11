# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""
Sequential Skeptic strategies against upper-expectation (imprecise) Forecasters.

Based on these papers:

Game-Theoretic Foundations for Probability and Finance, G. Shafer, V. Vovk (2019), Wiley
    - Chapter 6 (protocols in which Forecaster announces an upper expectation)

Hypothesis testing with e-values, A. Ramdas, R. Wang (2025) - https://arxiv.org/pdf/2410.23614
    - Definition 7.21(i) (predictable betting on sequential e-values), Ville's inequality
    - Proposition 2.4 (anytime-valid p-value)

Universal Prediction of Individual Sequences, Krichevsky-Trofimov estimator
    - R. E. Krichevsky, V. K. Trofimov (1981), IEEE Trans. Inf. Theory 27(2):
      the add-1/2 predictive law used as Skeptic's predictable belief

The numeraire e-variable and reverse information projection, M. Larsson, A. Ramdas,
J. Ruf (2025) - https://arxiv.org/pdf/2402.18810
    - Definition 2.1, Theorem 3.6: the numeraire is the optimal Kelly bet against P under Q

Protocol.  At round t Forecaster announces a convex set P_t of laws on a finite outcome
space (a fixed set, or a new one each round, e.g. a prediction market's bid-ask spread),
Reality announces x_t, and Skeptic pays out e_t(x_t), where e_t is the certified optimal
e-variable for P_t against Skeptic's *predictable* belief q_{t-1}.  Because e_t is
F_{t-1}-measurable and sup_{P in P_t} <P, e_t> <= 1 by certificate,
E[e_t(x_t) | F_{t-1}] <= 1 under any adapted choice of P_t in P_t: Reality may pick a
different member of the set every round after seeing the past.  Temporal betting is
delegated to ``BettingProtocol``.

``WassersteinDistanceSequence`` inverts the Wasserstein-ball Skeptic over a grid of radii.
The ball for the true distance rho* is a true null, so the test at the first grid radius
>= rho* is anytime valid, and the smallest non-rejected radius is a lower confidence bound
on W1(Reality, P0) that holds uniformly over time (up to one grid step).  All radii share
one Kantorovich potential; only the tax differs.
"""

from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from expectation.modules.eprocessupdater import EProcessUpdater
from expectation.modules.hypothesistesting import EProcess, EProcessConfig
from expectation.modules.protocol import SkepticStrategy
from expectation.upperexp.nulls import UpperExpectationNull, WassersteinNull, growth_rate


class UpperExpectationConfig(BaseModel):
    """
    Configuration for ``UpperExpectationSkeptic``.

    Parameters
    ----------
    smoothing : float
        Pseudo-count added to every outcome in Skeptic's predictable belief
        q_{t-1} = (counts + smoothing) / (n + K smoothing); 0.5 is Krichevsky-Trofimov.
    alternative : list of float, optional
        A fixed belief q instead of the plug-in; the e-variable is then the static
        GRO / numeraire e-variable against q.
    """

    smoothing: float = Field(default=0.5, gt=0)
    alternative: Optional[list[float]] = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_alternative(self) -> "UpperExpectationConfig":
        if self.alternative is not None:
            if any(a < 0 for a in self.alternative):
                raise ValueError("alternative must be nonnegative")
            if abs(sum(self.alternative) - 1.0) > 1e-8:
                raise ValueError("alternative must sum to one")
        return self


class DistanceSequenceResult(BaseModel):
    """
    One step of ``WassersteinDistanceSequence``.

    Parameters
    ----------
    step : int
        Round index (1-based).
    lower_bound : float
        Anytime-valid lower confidence bound on W1(Reality, P0): the smallest grid radius
        whose ball has not been rejected (the largest radius if all were).
    all_rejected : bool
        Whether every radius on the grid has been rejected.
    rejected_radii : list of float
        Radii rejected so far (running maximum of that ball's e-process >= 1/alpha).
    log_e_processes : list of float
        Current log capital for each radius on the grid.
    wasserstein_estimate : float
        Plug-in W1(q_{t-1}, P0) used for this step's bets (diagnostic only).
    """

    step: int = Field(ge=1)
    lower_bound: float = Field(ge=0)
    all_rejected: bool
    rejected_radii: list[float]
    log_e_processes: list[float]
    wasserstein_estimate: float = Field(ge=0)

    model_config = ConfigDict(frozen=True)


class UpperExpectationSkeptic(SkepticStrategy):
    """
    Certified log-optimal betting against a convex-set Forecaster.

    Parameters
    ----------
    null : UpperExpectationNull, optional
        Forecaster's fixed announcement.  May be omitted when every round supplies a
        null through the ``announcement`` argument (then ``config.alternative`` or the
        first announcement fixes the number of outcomes).
    config : UpperExpectationConfig, optional
        Belief smoothing / fixed alternative.
    """

    name = "upper_expectation"

    def __init__(
        self,
        null: Optional[UpperExpectationNull] = None,
        config: Optional[UpperExpectationConfig] = None,
    ):
        self.null = null
        self.config = config or UpperExpectationConfig()
        self.n_outcomes: Optional[int] = None
        if null is not None:
            self.n_outcomes = null.n_outcomes
        if self.config.alternative is not None:
            if self.n_outcomes is not None and len(self.config.alternative) != self.n_outcomes:
                raise ValueError("alternative length must equal the null's number of outcomes")
            self.n_outcomes = len(self.config.alternative)
        self.counts: Optional[NDArray] = (
            None if self.n_outcomes is None else np.zeros(self.n_outcomes)
        )
        self.n_observations: int = 0
        self._last: dict[str, Any] = {}
        self._cache_key: Optional[tuple] = None
        self._cache_e: Optional[NDArray] = None

    def _resolve_null(self, announcement: Any) -> UpperExpectationNull:
        null = announcement if announcement is not None else self.null
        if null is None:
            raise ValueError("No null announced: pass a fixed null or a per-round announcement")
        if not isinstance(null, UpperExpectationNull):
            raise TypeError("announcement must be an UpperExpectationNull")
        if self.n_outcomes is None:
            self.n_outcomes = null.n_outcomes
            self.counts = np.zeros(self.n_outcomes)
        elif null.n_outcomes != self.n_outcomes:
            raise ValueError("announced null has a different number of outcomes")
        return null

    def belief(self) -> NDArray:
        """Skeptic's predictable belief q_{t-1} (Krichevsky-Trofimov smoothed counts)."""
        if self.config.alternative is not None:
            return np.asarray(self.config.alternative, dtype=np.float64)
        if self.counts is None:
            raise ValueError("number of outcomes unknown until a null is announced")
        K = len(self.counts)
        return (self.counts + self.config.smoothing) / (
            self.n_observations + K * self.config.smoothing
        )

    def e_variable(self, announcement: Any = None) -> NDArray:
        """The certified e-variable vector Skeptic uses this round (cached per belief)."""
        null = self._resolve_null(announcement)
        q = self.belief()
        key = (id(null), self.n_observations, q.tobytes())
        if self._cache_key == key and self._cache_e is not None:
            return self._cache_e
        e = null.optimal_e_variable(q)
        self._cache_key, self._cache_e = key, e
        self._last = {
            "growth_estimate": growth_rate(q, e),
            "certificate": null.last_certificate,
            "belief_in_null": bool(np.allclose(e, 1.0)),
        }
        return e

    def _check_outcome(self, reality_move: Any) -> int:
        x = int(reality_move)
        if self.n_outcomes is None or not 0 <= x < self.n_outcomes:
            raise ValueError(f"outcome index {x} out of range [0, {self.n_outcomes})")
        return x

    def bet(self, reality_move: Any, announcement: Any = None) -> float:
        e = self.e_variable(announcement)
        return float(e[self._check_outcome(reality_move)])

    def observe(self, reality_move: Any, announcement: Any = None) -> None:
        self._resolve_null(announcement)
        x = self._check_outcome(reality_move)
        assert self.counts is not None
        self.counts[x] += 1
        self.n_observations += 1

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last)

    def reset(self) -> None:
        if self.counts is not None:
            self.counts = np.zeros_like(self.counts)
        self.n_observations = 0
        self._last = {}
        self._cache_key = None
        self._cache_e = None


class WassersteinDistanceSequence:
    """
    Anytime-valid lower confidence bound on W1(Reality, P0) on a finite metric space.

    Parameters
    ----------
    p0 : array_like
        Reference law (strictly positive).
    cost_matrix : array_like
        Ground metric.
    radii : sequence of float
        Strictly increasing grid of candidate radii rho.
    significance_level : float
        alpha for Ville's inequality on each radius' e-process.
    config : UpperExpectationConfig, optional
        Belief smoothing / fixed alternative.

    Notes
    -----
    Each radius owns an ``EProcess`` driven by the library's ``EProcessUpdater`` with
    all-in betting (the bet size is already log-optimised inside the witness), so
    rejection and stopping times are the library's own Ville logic.  Coverage: for the
    first grid radius rho_j >= W1(Reality, P0), P(rho_j ever rejected) <= alpha, hence
    P(lower_bound <= rho_j for all t) >= 1 - alpha.
    """

    def __init__(
        self,
        p0: NDArray,
        cost_matrix: NDArray,
        radii: Sequence[float],
        significance_level: float = 0.05,
        config: Optional[UpperExpectationConfig] = None,
    ):
        radii_arr = np.asarray(radii, dtype=np.float64)
        if radii_arr.ndim != 1 or len(radii_arr) == 0:
            raise ValueError("radii must be a non-empty 1-D sequence")
        if np.any(radii_arr < 0) or np.any(np.diff(radii_arr) <= 0):
            raise ValueError("radii must be nonnegative and strictly increasing")
        self.radii = radii_arr
        self.config = config or UpperExpectationConfig()
        self.e_process_config = EProcessConfig(significance_level=significance_level)
        self.ball = WassersteinNull(p0, cost_matrix, radius=float(radii_arr[0]))
        self.skeptic = UpperExpectationSkeptic(self.ball, self.config)
        self.e_process_updater = EProcessUpdater(self.e_process_config)
        self.e_processes = [EProcess(config=self.e_process_config) for _ in self.radii]
        self.step: int = 0
        self.history: list[dict[str, Any]] = []

    def update(self, reality_move: int) -> DistanceSequenceResult:
        x = int(reality_move)
        if not 0 <= x < self.ball.n_outcomes:
            raise ValueError(f"outcome index {x} out of range [0, {self.ball.n_outcomes})")
        q = self.skeptic.belief()
        h = self.ball.kantorovich_potential(q)
        estimate = float((q - self.ball.p0) @ h)
        e_values = [self.ball.witness_e_variable(h, q, radius=rho)[x] for rho in self.radii]

        # commit (all bets were F_{t-1}-measurable; no state changed above)
        self.skeptic.observe(x)
        for process, e_value in zip(self.e_processes, e_values):
            self.e_process_updater.update(process, float(e_value))
        self.step += 1

        rejected = np.array(
            [self.e_process_updater.is_significant(p) for p in self.e_processes], dtype=bool
        )
        if np.all(rejected):
            lower, all_rejected = float(self.radii[-1]), True
        else:
            lower, all_rejected = float(self.radii[~rejected][0]), False
        log_e = [float(p.log_process_values[-1]) for p in self.e_processes]
        result = DistanceSequenceResult(
            step=self.step,
            lower_bound=lower,
            all_rejected=all_rejected,
            rejected_radii=self.radii[rejected].tolist(),
            log_e_processes=log_e,
            wasserstein_estimate=max(estimate, 0.0),
        )
        self.history.append(
            {"step": self.step, "lower_bound": lower, "wasserstein_estimate": estimate}
        )
        return result

    def reset(self) -> None:
        self.skeptic.reset()
        self.e_process_updater = EProcessUpdater(self.e_process_config)
        self.e_processes = [EProcess(config=self.e_process_config) for _ in self.radii]
        self.step = 0
        self.history = []
