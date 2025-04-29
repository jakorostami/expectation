from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import re
import json
from expectation.agentic.models import TestProposal, TestState, TestConfig, HypothesisTestType, TestDirection
from expectation.agentic.agents import DesignerAgent


class RuleBasedDesignerAgent(DesignerAgent):
    """
    Rule based implementation of the designer agent.

    This one uses predefined rules to propose tests based on the hypothesis statement and available data.
    """

    def __init__(self):

        self.test_patterns = [
                (r"mean|average|expectation", self._propose_mean_test),
                (r"correlation|relationship|association", self._propose_correlation_test),
                (r"proportion|percentage|frequency|probability", self._propose_proportion_test),
                (r"variance|difference|variation", self._propose_variance_test),
            ]

    def process(self, state: TestState, data: Dict[str, pd.DataFrame]) -> TestProposal:

        hypothesis = state.hypothesis.lower()

        for pattern, proposal_method in self.test_patterns:
            if re.search(pattern, hypothesis):
                return proposal_method(hypothesis, data, state.proposals)
        
        return self._propose_default_test(hypothesis, data, state.proposals)
    
    def _propose_mean_test(self, hypothesis: str, data: Dict[str, pd.DataFrame], existing_proposals: List[TestProposal]) -> TestProposal:

        numeric_vars = []
        for df_name, df in data.items():
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    numeric_vars.append(df_name, col)


        if not numeric_vars:
            raise ValueError("No numeric variables found in the data.")
        
        used_vars = set()
        for proposal in existing_proposals:
            used_vars.update(proposal.required_variables)

        

