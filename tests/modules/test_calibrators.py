import numpy as np
import pytest

from expectation.modules.calibrators import EToPCalibrator, PToECalibrator, PToECalibratorType


class TestEToPCalibrator:
    @pytest.fixture
    def calibrator(self):
        return EToPCalibrator()

    def test_basic_conversion(self, calibrator):
        assert calibrator(20) == 0.05
        assert calibrator(10) == 0.1
        assert calibrator(1.0) == 1.0

    def test_array_input(self, calibrator):
        e_values = np.array([10, 20, 100])
        expected = np.array([0.1, 0.05, 0.01])
        result = calibrator(e_values)
        np.testing.assert_allclose(result, expected)

    def test_boundary_cases(self, calibrator):
        # e < 1 should give p = 1
        assert calibrator(0.5) == 1.0

        # large e should give small p
        assert calibrator(1000) == 0.001


class TestPToECalibrator:
    def test_shafer_default(self):
        calibrator = PToECalibrator()
        assert calibrator(0.01) == pytest.approx(9.0)  # page 24 chapter 2
        assert calibrator(0.25) == pytest.approx(1.0)

    def test_linear(self):
        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.LINEAR)
        assert calibrator(0.01) == pytest.approx(1.98)
        assert calibrator(0.5) == pytest.approx(1.0)
        assert calibrator(0.0) == pytest.approx(2.0)
        assert calibrator(1.0) == pytest.approx(0.0)

    def test_power(self):
        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.POWER, kappa=0.5)
        assert calibrator(0.01) == pytest.approx(5.0, rel=1e-6)

    def test_logarithmic(self):
        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.LOGARITHMIC)
        assert calibrator(0.01) == pytest.approx(-np.log(0.01))

    def test_mixture(self):
        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.MIXTURE)
        result = calibrator(0.1)
        assert result > 0
        assert np.isfinite(result)

    def test_kappa_validation(self):
        PToECalibrator(calibrator_type=PToECalibratorType.POWER, kappa=0.5)

        with pytest.raises(ValueError):
            PToECalibrator(calibrator_type=PToECalibratorType.POWER, kappa=1.5)

    def test_array_input(self):
        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.SHAFER)
        p_values = np.array([0.01, 0.04, 0.25])
        expected = np.array([9.0, 4.0, 1.0])
        result = calibrator(p_values)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_edge_cases(self):
        # p = 0 tests (unbounded calibrators should give infinity)
        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.POWER, kappa=0.5)
        assert np.isinf(calibrator(0.0))

        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.SHAFER)
        assert np.isinf(calibrator(0.0))

        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.LOGARITHMIC)
        assert np.isinf(calibrator(0.0))

        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.MIXTURE)
        assert np.isinf(calibrator(0.0))

        # linear is bounded at p=0
        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.LINEAR)
        assert calibrator(0.0) == pytest.approx(2.0)

        # p = 1 tests
        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.POWER, kappa=0.5)
        assert calibrator(1.0) == pytest.approx(0.5)  # kappa * 1^(kappa-1) = kappa

        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.LINEAR)
        assert calibrator(1.0) == pytest.approx(0.0)

        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.SHAFER)
        assert calibrator(1.0) == pytest.approx(0.0)

        calibrator = PToECalibrator(calibrator_type=PToECalibratorType.LOGARITHMIC)
        assert calibrator(1.0) == pytest.approx(0.0)

    def test_monotonicity(self):
        for cal_type in PToECalibratorType:
            calibrator = PToECalibrator(calibrator_type=cal_type, kappa=0.5)
            p_values = np.linspace(0.01, 0.99, 50)
            e_values = calibrator(p_values)
            assert np.all(np.diff(e_values) <= 1e-10)


class TestRoundTrip:
    def test_e_to_p_to_e(self):
        e_to_p = EToPCalibrator()
        p_to_e = PToECalibrator(calibrator_type=PToECalibratorType.SHAFER)

        p_values = np.linspace(0.01, 0.99, 20)
        for p in p_values:
            e = p_to_e(p)
            p_reconstructed = e_to_p(e)
            assert p_reconstructed >= p or np.isclose(p_reconstructed, p, rtol=1e-10)

    def test_p_to_e_to_p(self):
        e_to_p = EToPCalibrator()
        p_to_e = PToECalibrator(calibrator_type=PToECalibratorType.SHAFER)

        e_values = np.logspace(0, 2, 20)
        for e in e_values:
            p = e_to_p(e)
            e_reconstructed = p_to_e(p)
            assert e_reconstructed <= e + 1e-8
