from typing import Dict, List
import pandas as pd
import re
from expectation.agentic.models import (
    TestProposal, TestState, TestFamily, 
    SequentialTestType, SymmetryType, AlternativeType
)
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

        df_name, variable = next(
            ((df, var) for df, var in numeric_vars if var not in used_vars), numeric_vars[0]
        )

        alternative = AlternativeType.TWO_SIDED
        if re.search(r"increase|higher|greater|more|positive", hypothesis):
            alternative = AlternativeType.GREATER
        elif re.search(r"decrease|lower|less|negative", hypothesis):
            alternative = AlternativeType.LESS
        
        return TestProposal(
            name=f"Sequential e-test for mean of {variable}",
            description=f"Test whether the mean of {variable} is different from zero using sequential e-values",
            test_family=TestFamily.SEQUENTIAL_E_TEST,
            test_type=SequentialTestType.MEAN,
            null_hypothesis=f"The mean of {variable} is equal to zero",
            alternative_hypothesis=f"The mean of {variable} is {'not equal to' if alternative == AlternativeType.TWO_SIDED else 'greater than' if alternative == AlternativeType.GREATER else 'less than'} zero",
            alternative=alternative,
            null_value=0.0,
            required_variables=[variable]
        )
    
    def _propose_symmetry_test(
        self, 
        hypothesis: str, 
        data: Dict[str, pd.DataFrame],
        existing_proposals: List[TestProposal]
    ) -> TestProposal:
        numeric_vars = []
        for df_name, df in data.items():
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    numeric_vars.append((df_name, col))
        
        if not numeric_vars:
            raise ValueError("No numeric variables found for symmetry test")
        
        used_vars = set()
        for proposal in existing_proposals:
            used_vars.update(proposal.required_variables)
        
        df_name, variable = next(
            ((df, var) for df, var in numeric_vars if var not in used_vars),
            numeric_vars[0] 
        )
        
        used_symmetry_types = [
            p.test_type for p in existing_proposals 
            if p.test_family == TestFamily.SYMMETRY_TEST and p.test_type is not None
        ]
        
        if SymmetryType.FISHER not in used_symmetry_types:
            symmetry_type = SymmetryType.FISHER
        elif SymmetryType.SIGN not in used_symmetry_types:
            symmetry_type = SymmetryType.SIGN
        else:
            symmetry_type = SymmetryType.WILCOXON
        
        return TestProposal(
            name=f"{symmetry_type.value.title()} symmetry test for {variable}",
            description=f"Test if {variable} has a symmetric distribution around zero using the {symmetry_type.value} symmetry test",
            test_family=TestFamily.SYMMETRY_TEST,
            test_type=symmetry_type,
            null_hypothesis=f"The distribution of {variable} is symmetric around zero",
            alternative_hypothesis=f"The distribution of {variable} is not symmetric around zero",
            alternative=AlternativeType.TWO_SIDED,
            required_variables=[variable]
        )
    
    def _propose_proportion_test(
        self, 
        hypothesis: str, 
        data: Dict[str, pd.DataFrame],
        existing_proposals: List[TestProposal]
    ) -> TestProposal:
        
        binary_vars = []
        for df_name, df in data.items():
            for col in df.columns:
                if pd.api.types.is_bool_dtype(df[col]) or (
                    pd.api.types.is_numeric_dtype(df[col]) and 
                    df[col].dropna().isin([0, 1]).all()
                ):
                    binary_vars.append((df_name, col))
        
        if not binary_vars:
            for df_name, df in data.items():
                for col in df.columns:
                    if pd.api.types.is_categorical_dtype(df[col]) or (
                        not pd.api.types.is_numeric_dtype(df[col]) and 
                        df[col].nunique() <= 5
                    ):
                        binary_vars.append((df_name, col))
        
        if not binary_vars:
            raise ValueError("No suitable binary/categorical variables found for proportion test")
        
        used_vars = set()
        for proposal in existing_proposals:
            used_vars.update(proposal.required_variables)
        
        df_name, variable = next(
            ((df, var) for df, var in binary_vars if var not in used_vars),
            binary_vars[0]
        )
        
        null_value = 0.5  # Default null hypothesis (proportion = 0.5)
        alternative = AlternativeType.TWO_SIDED
        if re.search(r"increase|higher|greater|more|positive", hypothesis):
            alternative = AlternativeType.GREATER
        elif re.search(r"decrease|lower|less|negative", hypothesis):
            alternative = AlternativeType.LESS
        
        return TestProposal(
            name=f"Sequential proportion test for {variable}",
            description=f"Test whether the proportion of {variable} differs from {null_value}",
            test_family=TestFamily.SEQUENTIAL_E_TEST,
            test_type=SequentialTestType.PROPORTION,
            null_hypothesis=f"The proportion of {variable} is equal to {null_value}",
            alternative_hypothesis=f"The proportion of {variable} is {'not equal to' if alternative == AlternativeType.TWO_SIDED else 'greater than' if alternative == AlternativeType.GREATER else 'less than'} {null_value}",
            alternative=alternative,
            null_value=null_value,
            required_variables=[variable]
        )
    
    def _propose_variance_test(
        self, 
        hypothesis: str, 
        data: Dict[str, pd.DataFrame],
        existing_proposals: List[TestProposal]
    ) -> TestProposal:

        numeric_vars = []
        for df_name, df in data.items():
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_vars.append((df_name, col))
        
        if not numeric_vars:
            raise ValueError("No numeric variables found for variance test")
        
        used_vars = set()
        for proposal in existing_proposals:
            used_vars.update(proposal.required_variables)
        
        df_name, variable = next(
            ((df, var) for df, var in numeric_vars if var not in used_vars),
            numeric_vars[0]
        )
        
        null_value = 1.0  # Default null hypothesis (variance = 1)
        alternative = AlternativeType.TWO_SIDED
        if re.search(r"increase|higher|greater|more|variable", hypothesis):
            alternative = AlternativeType.GREATER
        elif re.search(r"decrease|lower|less|stable", hypothesis):
            alternative = AlternativeType.LESS
        
        return TestProposal(
            name=f"Sequential variance test for {variable}",
            description=f"Test whether the variance of {variable} differs from {null_value}",
            test_family=TestFamily.SEQUENTIAL_E_TEST,
            test_type=SequentialTestType.VARIANCE,
            null_hypothesis=f"The variance of {variable} is equal to {null_value}",
            alternative_hypothesis=f"The variance of {variable} is {'not equal to' if alternative == AlternativeType.TWO_SIDED else 'greater than' if alternative == AlternativeType.GREATER else 'less than'} {null_value}",
            alternative=alternative,
            null_value=null_value,
            required_variables=[variable]
        )
    
    def _propose_default_test(
        self, 
        hypothesis: str, 
        data: Dict[str, pd.DataFrame],
        existing_proposals: List[TestProposal]
    ) -> TestProposal:
        for df_name, df in data.items():
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col].dtype):
                    return TestProposal(
                        name=f"Universal t-test for {col}",
                        description=f"Test whether the mean of {col} is different from zero using universal t-test",
                        test_family=TestFamily.UNIVERSAL_T_TEST,
                        null_hypothesis=f"The mean of {col} is equal to zero",
                        alternative_hypothesis=f"The mean of {col} is not equal to zero",
                        alternative=AlternativeType.TWO_SIDED,
                        null_value=0.0,
                        required_variables=[col]
                    )
        
        # If no numeric variables, try symmetry test
        for df_name, df in data.items():
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col].dtype):
                    return TestProposal(
                        name=f"Fisher symmetry test for residuals",
                        description=f"Create residuals from {col} and test for symmetry",
                        test_family=TestFamily.SYMMETRY_TEST,
                        test_type=SymmetryType.FISHER,
                        null_hypothesis=f"The residuals from {col} are symmetrically distributed",
                        alternative_hypothesis=f"The residuals from {col} are not symmetrically distributed",
                        alternative=AlternativeType.TWO_SIDED,
                        required_variables=[col]
                    )
        
        raise ValueError("No suitable variables found for any test")

        

class LLMDesignerAgent(DesignerAgent):
    """
    LLM-based designer agent.
    
    Uses a language model to propose statistical tests based on
    the hypothesis and data context.
    """
    
    def __init__(self, config: AgentConfig, llm_provider: str = "openai", model_name: str = "gpt-4"):
        super().__init__(config)
        self.llm_provider = llm_provider
        self.model_name = model_name
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM client."""
        if self.llm_provider == "openai":
            import openai
            self.client = openai.OpenAI()
        elif self.llm_provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def process(self, state: TestState, data: Dict[str, pd.DataFrame]) -> TestProposal:
        """
        Propose a test using LLM.
        
        Args:
            state: Current test state
            data: Dictionary of dataframes
            
        Returns:
            A test proposal
        """
        # Create data context
        data_context = self._create_data_context(data)
        
        # Create prompt
        prompt = self._create_proposal_prompt(state, data_context)
        
        # Generate response from LLM and parse to TestProposal
        # Implementation details would depend on the specific LLM API
        pass
    
    def _create_data_context(self, data: Dict[str, pd.DataFrame]) -> str:
        """Create a description of the data for context."""
        # Creates a text description of the dataframes and their contents
        pass
    
    def _create_proposal_prompt(self, state: TestState, data_context: str) -> str:
        """Create a prompt for the LLM."""
        # Creates a prompt asking for a test proposal
        pass