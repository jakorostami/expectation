from expectation.agentic.models import AgentConfig
from expectation.agentic.designer import RuleBasedDesignerAgent, LLMDesignerAgent
from expectation.agentic.executor import StandardExecutorAgent
from expectation.agentic.evaluator import StandardEvaluatorAgent
from expectation.agentic.orchestrator import HypothesisTestOrchestrator
from typing import Dict, Any
import pandas as pd

class SequentialFalsificationTester:
    """
    High-level API for sequential falsification testing.
    
    This class provides a user-friendly interface for testing
    scientific hypotheses using sequential falsification testing.
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        max_iterations: int = 5,
        timeout: float = 60.0,
        domain: str = "general",
        combine_method: str = "product",
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
            combine_method=combine_method
        )
        
        # Create agents based on configuration
        if use_llm:
            self.designer_agent = LLMDesignerAgent(
                self.config, 
                llm_provider=llm_provider, 
                model_name=llm_model
            )
        else:
            self.designer_agent = RuleBasedDesignerAgent(self.config)
            
        self.executor_agent = StandardExecutorAgent(self.config)
        self.evaluator_agent = StandardEvaluatorAgent(self.config)
        
        # Create orchestrator
        self.orchestrator = HypothesisTestOrchestrator(
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
        Test a hypothesis using sequential falsification.
        
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
        
        return {
            "hypothesis": final_state.hypothesis,
            "conclusion": final_state.conclusion,
            "confidence": final_state.confidence,
            "combined_e_value": final_state.combined_e_value,
            "num_tests": len(final_state.results),
            "test_summary": test_summary,
            "iterations": final_state.iteration
        }