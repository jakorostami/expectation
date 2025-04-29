from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Protocol, TypeVar
import pandas as pd
import numpy as np
from expectation.agentic.models import (
    TestProposal,
    TestResult,
    TestState,
    TestConfig,
)

T = TypeVar("T")

class Agent(ABC, Protocol[T]):

    """
    Base protocol for all agents
    """

    @abstractmethod
    def process(self, state: TestState, data: Dict[str, pd.DataFrame]) -> T:
        pass

class DesignerAgent(Agent[TestProposal]):
    """
    Agent that proposes new tests based on the current state and data.
    """

    def __init__(self, config: TestConfig):
        self.config = config

    @abstractmethod
    def process(self, state: TestState, data: Dict[str, pd.DataFrame]) -> TestProposal:
        pass


class ExecutionAgent(Agent[TestResult]):
    """
    Agent that executes tests based on the current state and data.
    """

    def __init__(self, config: TestConfig):
        self.config = config

    @abstractmethod
    def process(self, state: TestState, data: Dict[str, pd.DataFrame]) -> TestResult:
        pass


class EvaluationAgent(Agent[Tuple[bool, float, str]]):
    """
    Agent responsible for evaluating the test results
    """

    def __init__(self, config: TestConfig):
        self.config = config

    @abstractmethod
    def process(self, state: TestState, data: Dict[str, pd.DataFrame]) -> Tuple[bool, float, str]:
        pass

