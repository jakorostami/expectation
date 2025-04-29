# expectation/agentic/__init__.py (updated with meta-analysis options)
from expectation.agentic.models import AgentConfig, ECalibrationMethod
from expectation.agentic.designer import RuleBasedDesignerAgent
from expectation.agentic.executor import StandardExecutorAgent
from expectation.agentic.evaluator import StandardEvaluatorAgent
from expectation.agentic.orchestrator import SequentialTestOrchestrator, ParallelTestOrchestrator
from typing import Dict, Any, Literal
import pandas as pd

class SequentialFalsificationTester:
    """
    High-level API for sequential falsification testing.
    
    This class provides a user-friendly interface for testing
    scientific hypotheses using sequential falsification testing
    with e-value meta-analysis capabilities.
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        max_iterations: int = 5,
        timeout: float = 60.0,
        domain: str = "general",
        combine_method: Literal["product", "fisher", "e_calibrator"] = "product",
        e_calibrator_method: Literal["kappa", "integral"] = "kappa",
        kappa: float = 0.5,
        parallel_testing: bool = False,
        max_parallel_tests: int = 3,
        use_llm: bool = False,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4"
    ):
        """
        Initialize the tester.
        
        Args:
            significance_level: Significance level for testing
            max_iterations: Maximum number of test iterations
            timeout: Timeout for test execution in seconds
            domain: Domain of the hypothesis
            combine_method: Method for combining e-values
            e_calibrator_method: Method for e-value calibration when using e_calibrator
            kappa: Kappa parameter for e-calibrator (when using kappa method)
            parallel_testing: Whether to run tests in parallel
            max_parallel_tests: Maximum number of parallel tests to run
            use_llm: Whether to use LLM-based agents
            llm_provider: Provider for LLM (if use_llm is True)
            llm_model: Model name for LLM (if use_llm is True)
        """
        # Create configuration
        self.config = AgentConfig(
            significance_level=significance_level,
            max_iterations=max_iterations,
            timeout=timeout,
            domain=domain,
            combine_method=combine_method,
            e_calibrator_method=ECalibrationMethod(e_calibrator_method),
            kappa=kappa,
            parallel_testing=parallel_testing,
            max_parallel_tests=max_parallel_tests
        )
        
        # Create agents
        self.designer_agent = RuleBasedDesignerAgent(self.config)
        self.executor_agent = StandardExecutorAgent(self.config)
        self.evaluator_agent = StandardEvaluatorAgent(self.config)
        
        # Create appropriate orchestrator
        if parallel_testing:
            self.orchestrator = ParallelTestOrchestrator(
                self.designer_agent,
                self.executor_agent,
                self.evaluator_agent,
                self.config
            )
        else:
            self.orchestrator = SequentialTestOrchestrator(
                self.designer_agent,
                self.executor_agent,
                self.evaluator_agent,
                self.config
            )
    
    def test(
        self, 
        hypothesis: str, 
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Test a hypothesis using sequential falsification with e-value meta-analysis.
        
        Args:
            hypothesis: The hypothesis to test
            data: Dictionary of dataframes for analysis
            
        Returns:
            Dictionary of test results
        """
        # Run the orchestrator
        final_state = self.orchestrator.test_hypothesis(hypothesis, data)
        
        # Format the results
        test_summary = []
        for i, result in enumerate(final_state.results):
            proposal = next((p for p in final_state.proposals if p.id == result.proposal_id), None)
            if not proposal:
                continue
                
            test_summary.append({
                "name": proposal.name,
                "description": proposal.description,
                "e_value": result.e_value,
                "p_value": result.p_value,
                "is_significant": result.is_significant,
                "interpretation": result.interpretation
            })
        
        # Get method description for clarity
        method_description = self.evaluator_agent._get_method_description()
        
        return {
            "hypothesis": final_state.hypothesis,
            "conclusion": final_state.conclusion,
            "confidence": final_state.confidence,
            "combined_e_value": final_state.combined_e_value,
            "combination_method": method_description,
            "num_tests": len(final_state.results),
            "test_summary": test_summary,
            "iterations": final_state.iteration,
            "parallel_testing": self.config.parallel_testing
        }