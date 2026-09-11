# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""
Forecaster-Skeptic-Reality testing protocol.

Based on these sources:

Game-Theoretic Foundations for Probability and Finance, G. Shafer, V. Vovk (2019), Wiley
    - Chapter 1 (the testing protocol: Forecaster, Skeptic, Reality; Skeptic's capital)
    - Chapter 8 (Ville's inequality for nonnegative supermartingales)

Testing by betting: a strategy for statistical and scientific communication,
G. Shafer (2021), JRSS-A 184(2) - https://doi.org/10.1111/rssa.12647
    - Section 2 (betting score = capital of a bet against the null)

Hypothesis testing with e-values, A. Ramdas, R. Wang (2025) - https://arxiv.org/pdf/2410.23614
    - Definition 7.21(i) (e-process from sequential e-values with predictable betting fractions)
    - Proposition 2.4 (anytime-valid p-value 1 / max_s M_s)

The library already owns Skeptic's *temporal* betting (``EProcessUpdater`` and the
``SequentialEValueCombiner`` strategies) and the *spatial* averaging of several
Skeptics (``merging.py``).  This module adds the missing half: ``SkepticStrategy``,
the object that turns Forecaster's announcement and Reality's move into the next
sequential e-value, and ``BettingProtocol``, the orchestrator that wires any such
strategy to the existing layers.  It mirrors ``KSampleSequentialTest`` once so new
strategies do not copy that orchestration.

Two-phase contract.  ``bet`` is pure with respect to strategy state and may only use
information committed through ``observe`` on earlier rounds, which is what makes the
e-value ``F_{t-1}``-measurable.  ``BettingProtocol.update`` validates the e-value
(finite, nonnegative) before touching the e-process and before calling ``observe``,
so a failed round leaves every component in its previous state.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from expectation.modules.calibrators import EToPCalibrator
from expectation.modules.eprocessupdater import EProcessUpdater
from expectation.modules.hypothesistesting import EProcess, EProcessConfig


class SkepticStrategy(ABC):
    """A strategy for Skeptic: Reality's moves in, sequential e-values out.

    Under Forecaster's announcement at round t the implementation must satisfy
    E[bet(X_t, announcement_t) | F_{t-1}] <= 1, where F_{t-1} is the information
    committed through ``observe`` on rounds 1..t-1.
    """

    name: str = "skeptic"

    @abstractmethod
    def bet(self, reality_move: Any, announcement: Any = None) -> float:
        """Sequential e-value for ``reality_move`` given the announcement (no state change)."""
        pass

    @abstractmethod
    def observe(self, reality_move: Any, announcement: Any = None) -> None:
        """Commit ``reality_move`` (and the announcement) to the strategy's state."""
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def diagnostics(self) -> dict[str, Any]:
        """Per-round diagnostics for the most recent ``bet`` call (optional)."""
        return {}


class ProtocolStepResult(BaseModel):
    """Result of one round of the testing protocol.

    Parameters
    ----------
    step : int
        Round index (1-based).
    e_value : float
        Sequential e-value emitted by Skeptic's strategy this round.
    e_process_value : float
        Skeptic's capital M_t after temporal betting.
    log_e_process : float
        log M_t.
    max_e_process : float
        Running maximum max_{s <= t} M_s, the quantity Ville's inequality bounds.
    p_value : float
        Anytime-valid p-value min(1, 1 / max_{s <= t} M_s) (Ramdas & Wang 2025, Prop. 2.4).
    reject_null : bool
        Whether max_{s <= t} M_s >= 1 / alpha.
    diagnostics : dict
        Strategy-specific diagnostics for this round.
    """

    step: int = Field(ge=1)
    e_value: float = Field(ge=0)
    e_process_value: float = Field(ge=0)
    log_e_process: float
    max_e_process: float = Field(ge=0)
    p_value: float = Field(ge=0, le=1)
    reject_null: bool
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class BettingProtocol:
    """Run a ``SkepticStrategy`` against a stream of Reality's moves.

    Parameters
    ----------
    strategy : SkepticStrategy
        Skeptic's strategy (emits sequential e-values).
    config : EProcessConfig, optional
        Temporal betting configuration (combiner and significance level); defaults to
        all-in betting at alpha = 0.05, i.e. capital is the running product of e-values.
    """

    def __init__(self, strategy: SkepticStrategy, config: Optional[EProcessConfig] = None):
        self.strategy = strategy
        self.config = config or EProcessConfig()
        self._calibrator = EToPCalibrator()

        self.e_process = EProcess(config=self.config)
        self.e_process_updater = EProcessUpdater(self.config)
        self.step: int = 0
        self.history: list[dict[str, Any]] = []

    @property
    def stopping_time(self) -> Optional[int]:
        return self.e_process_updater.get_stopping_time(self.e_process)

    def update(self, reality_move: Any, announcement: Any = None) -> ProtocolStepResult:
        """Play one round: Skeptic bets, Reality's move is scored, state is committed.

        Raises
        ------
        ValueError
            If the strategy emits a non-finite or negative e-value.  No state
            (e-process, history, strategy) is modified in that case.
        """
        e_value = float(self.strategy.bet(reality_move, announcement))
        if not np.isfinite(e_value) or e_value < 0:
            raise ValueError(
                f"Strategy {self.strategy.name!r} emitted an invalid e-value {e_value!r}; "
                "sequential e-values must be finite and nonnegative."
            )
        diagnostics = dict(self.strategy.diagnostics())

        self.e_process_updater.update(self.e_process, e_value)
        self.strategy.observe(reality_move, announcement)
        self.step += 1

        current = self.e_process_updater.get_current_value(self.e_process)
        running_max = self.e_process_updater.get_max_value(self.e_process)
        log_current = self.e_process.log_process_values[-1]
        reject_null = self.e_process_updater.is_significant(self.e_process)
        p_value = float(self._calibrator(running_max))

        result = ProtocolStepResult(
            step=self.step,
            e_value=e_value,
            e_process_value=float(current),
            log_e_process=float(log_current),
            max_e_process=float(running_max),
            p_value=p_value,
            reject_null=bool(reject_null),
            diagnostics=diagnostics,
        )
        self.history.append(
            {
                "step": self.step,
                "e_value": e_value,
                "e_process_value": float(current),
                "log_e_process": float(log_current),
                "max_e_process": float(running_max),
                "p_value": p_value,
                "reject_null": bool(reject_null),
                **{f"diag_{k}": v for k, v in diagnostics.items() if np.isscalar(v)},
            }
        )
        return result

    def get_history_df(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    def get_summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.name,
            "n_steps": self.step,
            "current_e_process": self.e_process_updater.get_current_value(self.e_process),
            "max_e_process": self.e_process_updater.get_max_value(self.e_process),
            "is_significant": self.e_process_updater.is_significant(self.e_process),
            "stopping_time": self.stopping_time,
            "empirical_growth_rate": self.e_process_updater.compute_asymptotic_growth_rate(
                self.e_process, min_samples=1
            ),
            "betting_strategy": self.config.betting_strategy.value,
        }

    def reset(self) -> None:
        self.strategy.reset()
        self.e_process = EProcess(config=self.config)
        self.e_process_updater = EProcessUpdater(self.config)
        self.step = 0
        self.history = []
