# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""Tests for the Forecaster-Skeptic-Reality testing protocol layer."""

import numpy as np
import pytest

from expectation.modules.hypothesistesting import BettingStrategy, EProcessConfig
from expectation.modules.protocol import BettingProtocol, ProtocolStepResult, SkepticStrategy


class ConstantSkeptic(SkepticStrategy):
    """Emits a fixed e-value; records which moves it observed."""

    name = "constant"

    def __init__(self, e_value: float):
        self._e = e_value
        self.observed: list = []

    def bet(self, reality_move, announcement=None) -> float:
        return self._e

    def observe(self, reality_move, announcement=None) -> None:
        self.observed.append(reality_move)

    def reset(self) -> None:
        self.observed = []


class TestBettingProtocol:
    def test_all_in_is_product_of_e_values(self):
        protocol = BettingProtocol(ConstantSkeptic(2.0))
        for step in range(1, 6):
            result = protocol.update(0.0)
            assert isinstance(result, ProtocolStepResult)
            assert result.step == step
            assert result.e_value == 2.0
            assert result.e_process_value == pytest.approx(2.0**step)
            assert result.log_e_process == pytest.approx(step * np.log(2.0))
        assert protocol.strategy.observed == [0.0] * 5

    def test_ville_rejection_and_p_value(self):
        protocol = BettingProtocol(
            ConstantSkeptic(2.0), config=EProcessConfig(significance_level=0.1)
        )
        results = [protocol.update(1.0) for _ in range(4)]
        # 2^3 = 8 < 10 <= 2^4 = 16
        assert [r.reject_null for r in results] == [False, False, False, True]
        assert results[-1].p_value == pytest.approx(1.0 / 16.0)
        assert protocol.stopping_time == 4

    def test_conservative_combiner_bets_fraction(self):
        config = EProcessConfig(
            betting_strategy=BettingStrategy.CONSERVATIVE, conservative_lambda=0.25
        )
        protocol = BettingProtocol(ConstantSkeptic(3.0), config=config)
        result = protocol.update(0.0)
        # (1 - lambda) + lambda * E = 0.75 + 0.75 = 1.5
        assert result.e_process_value == pytest.approx(1.5)

    @pytest.mark.parametrize("bad", [np.nan, -0.5, np.inf])
    def test_invalid_e_value_rejected_before_state_changes(self, bad):
        skeptic = ConstantSkeptic(bad)
        protocol = BettingProtocol(skeptic)
        with pytest.raises(ValueError):
            protocol.update(0.0)
        assert protocol.step == 0
        assert protocol.e_process.total_samples == 0
        assert skeptic.observed == []
        assert protocol.history == []

    def test_reset_is_fresh_experiment(self):
        skeptic = ConstantSkeptic(2.0)
        protocol = BettingProtocol(skeptic)
        for _ in range(3):
            protocol.update(0.0)
        protocol.reset()
        assert protocol.step == 0
        assert protocol.e_process.process_values == [1.0]
        assert skeptic.observed == []
        result = protocol.update(0.0)
        assert result.e_process_value == pytest.approx(2.0)
        assert result.step == 1

    def test_history_dataframe_and_summary(self):
        protocol = BettingProtocol(ConstantSkeptic(1.5))
        for _ in range(3):
            protocol.update(0.0)
        df = protocol.get_history_df()
        assert list(df["step"]) == [1, 2, 3]
        summary = protocol.get_summary()
        assert summary["strategy"] == "constant"
        assert summary["n_steps"] == 3
        assert summary["current_e_process"] == pytest.approx(1.5**3)

    def test_result_is_frozen(self):
        protocol = BettingProtocol(ConstantSkeptic(1.0))
        result = protocol.update(0.0)
        with pytest.raises(Exception):
            result.e_value = 5.0
