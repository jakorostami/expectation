from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, validator


class BoundaryType(str, Enum):
    NORMAL_MIXTURE = "normal_mixture"
    BETA_BINOMIAL = "beta_binomial" 
    GAMMA_EXPONENTIAL = "gamma_exponential"
    POLY_STITCHING = "poly_stitching"
    DISCRETE_MIXTURE = "discrete_mixture"

class EstimandType(str, Enum):
    MEAN = "mean"
    QUANTILE = "quantile"
    VARIANCE = "variance"
    PROPORTION = "proportion"

class ConfidenceSequenceConfig(BaseModel):
    alpha: float = Field(gt=0, lt=1, default=0.05)
    alpha_opt: float = Field(gt=0, lt=1, default=0.05)
    v_opt: float = Field(gt=0, default=1.0)
    c: float = Field(gt=0, default=1.0)  # Added default value
    boundary_type: BoundaryType = Field(default=BoundaryType.NORMAL_MIXTURE)
    model_config = ConfigDict(frozen=True)

# Updated EmpiricalBernsteinConfig with more robust defaults
class EmpiricalBernsteinConfig(ConfidenceSequenceConfig):
    """
    Configuration for Empirical Bernstein confidence sequences.
    
    Extends base config with bounds on the observations.
    """
    lower_bound: float = Field(description="Lower bound on observations")
    upper_bound: float = Field(description="Upper bound on observations")
    boundary_type: BoundaryType = Field(
        default=BoundaryType.NORMAL_MIXTURE,  # Changed default to more stable boundary
        description="Type of boundary to use"
    )
    rho: float = Field(
        default=2.0,  # Increased default for better stability
        gt=0,
        description="Tuning parameter controlling boundary shape"
    )
    
    @validator('upper_bound')
    def upper_bound_must_exceed_lower(cls, v: float, values: dict) -> float:
        if 'lower_bound' in values and v <= values['lower_bound']:
            raise ValueError('upper_bound must be greater than lower_bound')
        return v
