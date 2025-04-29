from pydantic import BaseModel, Field, ConfigDict
import uuid
from typing import List, Dict, Optional, Union, Literal
from enum import Enum
import numpy as np
from datetime import datetime

class HypothesisTestType(str, Enum):
    """Types of hypothesis tests available."""
    MEAN = "mean"
    PROPORTION = "proportion"
    QUANTILES = "quantiles"
    VARIANCE = "variance"
    SYMMETRIC = "symmetric"

class TestDirection(str, Enum):
    """Direction of the test."""
    TWO_SIDED = "two_sided"
    GREATER = "greater"
    LESS = "less"

class TestProposal(BaseModel):

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(description="Name of the test")
    description: str = Field(description="Detailed description of the test's purpose")
    test_type: HypothesisTestType = Field(description="Type of hypothesis test")
    null_hypothesis: str = Field(description="Formal null hypothesis statement")
    alternative_hypothesis: str = Field(description="Formal alternative hypothesis statement")
    direction: TestDirection = Field(description="Direction of the test")
    required_variables: List[str] = Field(description="List of variables required for the test")

    model_config = ConfigDict(frozen=True)
    

class TestResult(BaseModel):

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    proposal_id: str = Field(description="ID of the corresponding test proposal")
    p_value: Optional[float] = Field(description="P-value of the test result")
    e_value: float = Field(description="E-value of the test result")
    is_significant: bool = Field(description="Whether the result is statistically significant")
    raw_output: str = Field(description="Raw output of the test")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the test result")
    interpretation: str = Field(description="Human-readable interpretation of the test results")

    model_config = ConfigDict(frozen=True)

class TestState(BaseModel):

    hypothesis: str = Field(description="Hypothesis being tested")
    iteration: int = Field(default=0, description="Current iteration of the test")
    proposals: List[TestProposal] = Field(default_factory=list, description="List of test proposals")
    results: List[TestResult] = Field(default_factory=list, description="List of test results")
    current_proposal: Optional[TestProposal] = Field(default=None, description="Currently selected test proposal")
    e_values: List[float] = Field(default_factory=list, description="List of E-values from the tests")
    combined_e_value: float = Field(default=1.0, description="Product of all  e-values from tests")
    conclusion: Optional[bool] = Field(default=None, description="Final conclusion of the hypothesis test (True/False)")
    confidence: float = Field(default=0.0, description="Confidence level of the test result")

    model_config = ConfigDict(frozen=True)

class TestConfig(BaseModel):
    
    significance_level: float = Field(default=0.05, gt=0, lt=1)
    max_iterations: int = Field(default=5, gt=0)
    timeout: float = Field(default=60.0, gt=0, description="Timeout for test execution in seconds")
    domain: str = Field(default="general", description="Domain of the test (e.g., 'general', 'medical', etc.)")
    combine_method: Literal["product", "fisher", "e_calibrator"] = Field(default="product", description="Method for combining e-values")

    model_config = ConfigDict(frozen=True)





