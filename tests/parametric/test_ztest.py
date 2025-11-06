import numpy as np
import pandas as pd
import pytest

from expectation.parametric.ztest import SaviZtestStat, CheckAndReturnsEsMinParameterSide
from expectation.seqtest.sequential_e_testing import AlternativeType


class TestSaviZTestStat:

    def test_initialization(self):
        "Test class creation and computing n Eff"
        # One Sample
        test_stat = SaviZtestStat(z=3, parameter=0.4, n1 = 55)
        assert test_stat.parameter == 0.4
        assert test_stat.n1 == 55
        assert np.isfinite(test_stat.compute_n_eff())

        # Two Sample
        test_stat2 = SaviZtestStat(z=3, parameter=0.1,n1=39,n2=67)
        assert test_stat2.parameter == 0.1
        assert test_stat2.compute_n_eff() > 0

        # One sample and paired
        test_stat3 = SaviZtestStat(z=2.0, paired=True, n1 = 38, n2 = 41, parameter=0.3)
        assert test_stat3.n1 == 38
        assert test_stat3.n2 == 41
        assert test_stat3.compute_n_eff() == 38
    
    @pytest.mark.parametrize("etype", ["egauss", "ecauchy", "imom", "mom", "grow"])
    @pytest.mark.parametrize("alternative",["two_sided","greater","less"])
    def test_all_etype_compute(self, etype, alternative):
        "Test all e-value computation methods work"
        test_stat = SaviZtestStat(z=1.0, parameter=0.5, n1=100, etype=etype, alternative=alternative)
        result = test_stat.compute_e_value()

        if isinstance(result, tuple):
            assert np.isfinite(result[0]) and result[0] > 0
        else:
            assert np.isfinite(result) and result > 0
        
        # Two sample
        test_stat2 = SaviZtestStat(z=3.0, n1 = 100, n2 = 85, parameter=0.4, etype=etype, alternative=alternative)
        result = test_stat2.compute_e_value()

        if isinstance(result, tuple):
            assert np.isfinite(result[0]) and result[0] > 0
        else:
            assert np.isfinite(result) and result > 0

    @pytest.mark.parametrize("valid_es_name", ["noName", "meanDiffMin", "phiS","deltaMin", "deltaS",
                     "hrMin", "thetaS", "deltaTrue", "g", "kappaG"])
    @pytest.mark.parametrize("alternative", [AlternativeType.TWO_SIDED, AlternativeType.GREATER, AlternativeType.LESS])
    def test_parameter_valid(self, valid_es_name, alternative):
        result = CheckAndReturnsEsMinParameterSide(paramToCheck=0.5, 
                                                   es_min_name=valid_es_name, alternative=alternative)
        assert isinstance(result, (float, int))
        assert np.isfinite(result)
        # Invalid alternative should raise error
        with pytest.raises(ValueError):
            CheckAndReturnsEsMinParameterSide(paramToCheck=0.5, alternative="invalid", es_min_name="phi_s")
    
    @pytest.mark.parametrize("etype", ["ecauchy", "imom", "mom"])
    def test_integration_methods_return_tuples(self,etype):
        """Test methods that use integration return proper tuples."""
        test_stat = SaviZtestStat(z = 3.0, parameter=0.5, n1=100, etype=etype)
        result = test_stat.compute_e_value()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert np.isfinite(result[0]) and result[0] > 0  # E-value
        if result[1] is not None:
            assert np.isfinite(result[1]) # Error estimate
    '''
    def test_debug_mom_method(self):
        """Debug the mom method step by step."""
        test_stat = SaviZtestStat(z=3.0, parameter=0.5, n1=100, etype="mom")
    
        print(f"z: {test_stat.z}")
        print(f"parameter: {test_stat.parameter}")  
        print(f"nEff: {test_stat.nEff}")
        print(f"alternative: {test_stat.alternative}")
            
        try:
            result = test_stat.compute_mom()
            print(f"Result: {result}")
            print(f"Result type: {type(result)}")
            
            if isinstance(result, tuple):
                print(f"First element: {result[0]}, type: {type(result[0])}")
                print(f"Second element: {result[1]}, type: {type(result[1])}")
        except Exception as e:
            print(f"Error in compute_mom: {e}")

            '''