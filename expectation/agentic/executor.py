# expectation/agentic/executor.py
from typing import Dict, Optional
import pandas as pd
import numpy as np
import time
import io
import contextlib
import traceback
from .models import (
    TestProposal, TestResult, TestState,
    TestFamily, SymmetryType, AlternativeType
)
from expectation.agentic.agents import ExecutionAgent

# Import expectation's modules
from expectation.seqtest.sequential_e_testing import SequentialTest
from expectation.parametric.ttest_universal import create_ttest
from expectation.modules.hypothesistesting import SymmetryETest

class StandardExecutorAgent(ExecutionAgent):
    """
    Standard implementation of the executor agent.
    
    This agent executes statistical tests using expectation's built-in methods.
    """
    
    def process(self, state: TestState, data: Dict[str, pd.DataFrame]) -> TestResult:

        proposal = state.current_proposal
        if not proposal:
            raise ValueError("No current proposal to execute")
        
        # Execute the test based on its family and type
        if proposal.test_family == TestFamily.SEQUENTIAL_E_TEST:
            return self._execute_sequential_e_test(proposal, data)
        elif proposal.test_family == TestFamily.UNIVERSAL_T_TEST:
            return self._execute_universal_t_test(proposal, data)
        elif proposal.test_family == TestFamily.SYMMETRY_TEST:
            return self._execute_symmetry_test(proposal, data)
        else:
            raise ValueError(f"Unsupported test family: {proposal.test_family}")
    
    def _execute_sequential_e_test(self, proposal: TestProposal, data: Dict[str, pd.DataFrame]) -> TestResult:
        """Execute a sequential e-test."""
        # Extract the variable from the data
        variable = proposal.required_variables[0]
        df = self._find_dataframe_with_variable(data, variable)
        
        if df is None:
            raise ValueError(f"Variable {variable} not found in any dataframe")
        
        # Get the values
        values = df[variable].dropna().values
        
        # Use expectation's SequentialTest
        start_time = time.time()
        try:
            # Create and run the appropriate test type
            test_type = proposal.test_type.value if proposal.test_type else "mean"
            
            test = SequentialTest(
                test_type=test_type,
                null_value=proposal.null_value,
                alternative=proposal.alternative.value
            )
            
            result = test.update(values)
            
            # Capture output for raw output
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                print(f"Sequential e-test for {variable}")
                print(f"Test type: {test_type}")
                print(f"Number of observations: {len(values)}")
                
                if test_type == "mean":
                    print(f"Mean: {np.mean(values):.4f}")
                    print(f"Standard deviation: {np.std(values, ddof=1):.4f}")
                elif test_type == "proportion":
                    print(f"Proportion: {np.mean(values):.4f}")
                elif test_type == "variance":
                    print(f"Variance: {np.var(values, ddof=1):.4f}")
                
                print(f"P-value: {result.p_value:.4f}")
                print(f"E-value: {result.e_value:.4f}")
                print(f"Significant: {result.reject_null}")
            
            raw_output = output_buffer.getvalue()
            
            # Generate interpretation
            if result.reject_null:
                interpretation = f"The sequential e-test provides significant evidence (e-value: {result.e_value:.4f}) that the {test_type} of {variable} is {'different from' if proposal.alternative == AlternativeType.TWO_SIDED else 'greater than' if proposal.alternative == AlternativeType.GREATER else 'less than'} {proposal.null_value}."
            else:
                interpretation = f"The sequential e-test does not provide significant evidence (e-value: {result.e_value:.4f}) to reject the null hypothesis that the {test_type} of {variable} is equal to {proposal.null_value}."
            
            return TestResult(
                proposal_id=proposal.id,
                p_value=result.p_value,
                e_value=result.e_value,
                is_significant=result.reject_null,
                raw_output=raw_output,
                interpretation=interpretation
            )
            
        except Exception as e:
            error_msg = f"Error executing sequential e-test: {str(e)}\n{traceback.format_exc()}"
            return TestResult(
                proposal_id=proposal.id,
                p_value=1.0,
                e_value=1.0,
                is_significant=False,
                raw_output=error_msg,
                interpretation=f"The test failed with an error: {str(e)}"
            )
    
    def _execute_universal_t_test(self, proposal: TestProposal, data: Dict[str, pd.DataFrame]) -> TestResult:
        """Execute a universal t-test."""
        # Extract the variable from the data
        variable = proposal.required_variables[0]
        df = self._find_dataframe_with_variable(data, variable)
        
        if df is None:
            raise ValueError(f"Variable {variable} not found in any dataframe")
        
        # Get the values
        values = df[variable].dropna().values
        
        # Use expectation's universal t-test
        try:
            # Create the universal t-test
            ttest = create_ttest(
                null_value=proposal.null_value,
                alternative=proposal.alternative.value,
                method="universal_inference"
            )
            
            # Run the test
            result = ttest.update(values)
            
            # Capture output for raw output
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                print(f"Universal t-test for {variable}")
                print(f"Number of observations: {len(values)}")
                print(f"Mean: {np.mean(values):.4f}")
                print(f"Standard deviation: {np.std(values, ddof=1):.4f}")
                print(f"P-value: {result.p_value:.4f}")
                print(f"E-value: {result.e_value:.4f}")
                print(f"Significant: {result.reject_null}")
            
            raw_output = output_buffer.getvalue()
            
            # Generate interpretation
            if result.reject_null:
                interpretation = f"The universal t-test provides significant evidence (e-value: {result.e_value:.4f}) that the mean of {variable} is {'different from' if proposal.alternative == AlternativeType.TWO_SIDED else 'greater than' if proposal.alternative == AlternativeType.GREATER else 'less than'} {proposal.null_value}."
            else:
                interpretation = f"The universal t-test does not provide significant evidence (e-value: {result.e_value:.4f}) to reject the null hypothesis that the mean of {variable} is equal to {proposal.null_value}."
            
            return TestResult(
                proposal_id=proposal.id,
                p_value=result.p_value,
                e_value=result.e_value,
                is_significant=result.reject_null,
                raw_output=raw_output,
                interpretation=interpretation
            )
            
        except Exception as e:
            error_msg = f"Error executing universal t-test: {str(e)}\n{traceback.format_exc()}"
            return TestResult(
                proposal_id=proposal.id,
                p_value=1.0,
                e_value=1.0,
                is_significant=False,
                raw_output=error_msg,
                interpretation=f"The test failed with an error: {str(e)}"
            )
    
    def _execute_symmetry_test(self, proposal: TestProposal, data: Dict[str, pd.DataFrame]) -> TestResult:
        """Execute a symmetry test."""
        # Extract the variable from the data
        variable = proposal.required_variables[0]
        df = self._find_dataframe_with_variable(data, variable)
        
        if df is None:
            raise ValueError(f"Variable {variable} not found in any dataframe")
        
        # Get the values
        values = df[variable].dropna().values
        
        # Use expectation's symmetry test
        try:
            # Get test type
            test_type = SymmetryType.FISHER
            if proposal.test_type:
                test_type = proposal.test_type
            
            # Create the symmetry test
            test = SymmetryETest(test_type=test_type.value)
            
            # Run the test
            result = test.test(values)
            
            # Capture output for raw output
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                print(f"{test_type.value.title()} symmetry test for {variable}")
                print(f"Number of observations: {len(values)}")
                print(f"E-value: {result.value:.4f}")
                print(f"Significant: {result.significant}")
            
            raw_output = output_buffer.getvalue()
            
            # Generate interpretation
            if result.significant:
                interpretation = f"The {test_type.value} symmetry test provides significant evidence (e-value: {result.value:.4f}) that the distribution of {variable} is not symmetric around zero."
            else:
                interpretation = f"The {test_type.value} symmetry test does not provide significant evidence (e-value: {result.value:.4f}) to reject the null hypothesis that the distribution of {variable} is symmetric around zero."
            
            # Convert to standard result format
            p_value = 1.0 / result.value if result.value > 0 else 1.0
            
            return TestResult(
                proposal_id=proposal.id,
                p_value=p_value,
                e_value=result.value,
                is_significant=result.significant,
                raw_output=raw_output,
                interpretation=interpretation
            )
            
        except Exception as e:
            error_msg = f"Error executing symmetry test: {str(e)}\n{traceback.format_exc()}"
            return TestResult(
                proposal_id=proposal.id,
                p_value=1.0,
                e_value=1.0,
                is_significant=False,
                raw_output=error_msg,
                interpretation=f"The test failed with an error: {str(e)}"
            )
    
    def _find_dataframe_with_variable(self, data: Dict[str, pd.DataFrame], variable: str) -> Optional[pd.DataFrame]:
        """Find a dataframe containing the specified variable."""
        for df in data.values():
            if variable in df.columns:
                return df
        return None