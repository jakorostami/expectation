# expectation/agentic/models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union, Literal
from enum import Enum
import uuid
from datetime import datetime

class TestFamily(str, Enum):
    SEQUENTIAL_E_TEST = "sequential_e_test"
    UNIVERSAL_T_TEST = "universal_t_test"
    SYMMETRY_TEST = "symmetry_test"
    MARTINGALE_TEST = "martingale_test"
    E_PROCESS = "e_process"

class SequentialTestType(str, Enum):
    MEAN = "mean"
    PROPORTION = "proportion"
    VARIANCE = "variance"
    QUANTILE = "quantile"

class SymmetryType(str, Enum):
    FISHER = "fisher"
    SIGN = "sign"
    WILCOXON = "wilcoxon"

class AlternativeType(str, Enum):
    TWO_SIDED = "two_sided"
    GREATER = "greater"
    LESS = "less"

class TestProposal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Descriptive name of the test")
    description: str = Field(description="Detailed description of the test's purpose")
    test_family: TestFamily = Field(description="Family of the test")
    test_type: Optional[Union[SequentialTestType, SymmetryType, str]] = Field(
        default=None, description="Type of test within the family"
    )
    null_hypothesis: str = Field(description="Formal null hypothesis statement")
    alternative_hypothesis: str = Field(description="Formal alternative hypothesis statement")
    alternative: AlternativeType = Field(description="Alternative hypothesis type")
    null_value: Optional[float] = Field(default=0.0, description="Null hypothesis value")
    required_variables: List[str] = Field(description="Variables required for the test")
    
    model_config = ConfigDict(frozen=True)

class TestResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    proposal_id: str = Field(description="ID of the corresponding test proposal")
    p_value: Optional[float] = Field(description="P-value from the test (if available)")
    e_value: float = Field(description="E-value from the test")
    is_significant: bool = Field(description="Whether the result is statistically significant")
    raw_output: str = Field(description="Raw output from test execution")
    timestamp: datetime = Field(default_factory=datetime.now)
    interpretation: str = Field(description="Human-readable interpretation of the results")
    
    model_config = ConfigDict(frozen=True)

class TestState(BaseModel):
    hypothesis: str = Field(description="The scientific hypothesis being tested")
    iteration: int = Field(default=0, description="Current iteration number")
    proposals: List[TestProposal] = Field(default_factory=list, description="Test proposals")
    results: List[TestResult] = Field(default_factory=list, description="Test results")
    current_proposal: Optional[TestProposal] = Field(default=None)
    e_values: List[float] = Field(default_factory=list, description="E-values from tests")
    combined_e_value: float = Field(default=1.0, description="Product of all e-values")
    conclusion: Optional[bool] = Field(default=None, description="Final conclusion (True/False)")
    confidence: float = Field(default=0.0, description="Confidence in the conclusion")
    
    model_config = ConfigDict(frozen=True)

class AgentConfig(BaseModel):
    significance_level: float = Field(default=0.05, gt=0, lt=1)
    max_iterations: int = Field(default=5, gt=0)
    timeout: float = Field(default=60.0, gt=0, description="Timeout for test execution in seconds")
    domain: str = Field(default="general", description="Domain of the hypothesis")
    combine_method: Literal["product", "fisher", "e_calibrator"] = Field(
        default="product", 
        description="Method for combining e-values"
    )
    
    model_config = ConfigDict(frozen=True)