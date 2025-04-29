# expectation/agentic/orchestrator.py (updated with parallel capabilities)
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import concurrent.futures
from expectation.agentic.models import TestState, AgentConfig, TestResult
from expectation.agentic.agents import DesignerAgent, ExecutionAgent, EvaluationAgent

class HypothesisTestOrchestrator:
    """Base class for test orchestrators."""
    
    def __init__(
        self, 
        designer_agent: DesignerAgent,
        executor_agent: ExecutionAgent,
        evaluator_agent: EvaluationAgent,
        config: AgentConfig
    ):
        """
        Initialize the orchestrator.
        
        Args:
            designer_agent: Agent for proposing tests
            executor_agent: Agent for executing tests
            evaluator_agent: Agent for evaluating results
            config: Configuration for the testing process
        """
        self.designer_agent = designer_agent
        self.executor_agent = executor_agent
        self.evaluator_agent = evaluator_agent
        self.config = config
    
    def test_hypothesis(
        self, 
        hypothesis: str, 
        data: Dict[str, pd.DataFrame]
    ) -> TestState:
        """
        Test a hypothesis.
        
        Args:
            hypothesis: The hypothesis to test
            data: Dictionary of dataframes for analysis
            
        Returns:
            Final state after testing
        """
        raise NotImplementedError("Subclasses must implement this method")


class SequentialTestOrchestrator(HypothesisTestOrchestrator):
    """
    Orchestrates the sequential falsification testing workflow.
    
    This class manages the state transitions and coordinates the
    overall testing process, running one test at a time.
    """
    
    def test_hypothesis(
        self, 
        hypothesis: str, 
        data: Dict[str, pd.DataFrame]
    ) -> TestState:
        """
        Test a hypothesis using sequential falsification.
        
        Args:
            hypothesis: The hypothesis to test
            data: Dictionary of dataframes for analysis
            
        Returns:
            Final state after testing
        """
        # Initialize state
        state = TestState(hypothesis=hypothesis)
        
        # Main testing loop
        for iteration in range(1, self.config.max_iterations + 1):
            # Update iteration counter
            state = TestState(**{**state.model_dump(), "iteration": iteration})
            
            # Step 1: Propose a test
            try:
                proposal = self.designer_agent.process(state, data)
                # Update state with new proposal
                state = TestState(**{
                    **state.model_dump(), 
                    "proposals": [*state.proposals, proposal],
                    "current_proposal": proposal
                })
            except Exception as e:
                print(f"Error proposing test: {e}")
                continue
            
            # Step 2: Execute the test
            try:
                result = self.executor_agent.process(state, data)
                # Update state with new result
                state = TestState(**{
                    **state.model_dump(),
                    "results": [*state.results, result],
                    "e_values": [*state.e_values, result.e_value]
                })
            except Exception as e:
                print(f"Error executing test: {e}")
                continue
            
            # Step 3: Evaluate results
            try:
                conclusion, confidence, reasoning = self.evaluator_agent.process(state, data)
                # Update state with evaluation
                combined_e_value = self.evaluator_agent._combine_e_values(state.e_values)
                
                state = TestState(**{
                    **state.model_dump(),
                    "combined_e_value": combined_e_value,
                    "conclusion": conclusion,
                    "confidence": confidence
                })
            except Exception as e:
                print(f"Error evaluating results: {e}")
                continue
            
            # Check if we've reached a conclusion
            if conclusion is not None:
                break
        
        # If no conclusion was reached, make a final determination
        if state.conclusion is None:
            # Not enough evidence after max iterations
            state = TestState(**{
                **state.model_dump(),
                "conclusion": False,
            })
        
        return state


class ParallelTestOrchestrator(HypothesisTestOrchestrator):
    """
    Orchestrates parallel falsification testing.
    
    This class proposes multiple tests at once and executes them
    in parallel for more efficient hypothesis testing.
    """
    
    def test_hypothesis(
        self, 
        hypothesis: str, 
        data: Dict[str, pd.DataFrame]
    ) -> TestState:
        """
        Test a hypothesis using parallel falsification.
        
        Args:
            hypothesis: The hypothesis to test
            data: Dictionary of dataframes for analysis
            
        Returns:
            Final state after testing
        """
        # Initialize state
        state = TestState(hypothesis=hypothesis)
        
        # Determine how many tests to run
        num_tests = min(self.config.max_parallel_tests, self.config.max_iterations)
        
        # Main testing loop (each iteration is a batch of parallel tests)
        for batch in range(1, self.config.max_iterations + 1, num_tests):
            # Update iteration counter
            current_iteration = batch
            state = TestState(**{**state.model_dump(), "iteration": current_iteration})
            
            # Step 1: Propose multiple tests
            proposals = []
            for i in range(num_tests):
                try:
                    proposal = self.designer_agent.process(state, data)
                    proposals.append(proposal)
                    
                    # Update state for next proposal
                    state = TestState(**{
                        **state.model_dump(), 
                        "proposals": [*state.proposals, proposal]
                    })
                except Exception as e:
                    print(f"Error proposing test {i+1}: {e}")
            
            if not proposals:
                print("No valid test proposals were generated")
                break
            
            # Step 2: Execute tests in parallel
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_tests) as executor:
                # Submit each test for execution
                future_to_proposal = {
                    executor.submit(self._execute_test, TestState(**{
                        **state.model_dump(),
                        "current_proposal": proposal
                    }), data): proposal
                    for proposal in proposals
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_proposal):
                    proposal = future_to_proposal[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Test execution failed: {e}")
            
            if not results:
                print("No tests were successfully executed")
                continue
            
            # Step 3: Update state with all results
            e_values = [result.e_value for result in results]
            state = TestState(**{
                **state.model_dump(),
                "results": [*state.results, *results],
                "e_values": [*state.e_values, *e_values]
            })
            
            # Step 4: Evaluate combined results
            try:
                conclusion, confidence, reasoning = self.evaluator_agent.process(state, data)
                # Update state with evaluation
                combined_e_value = self.evaluator_agent._combine_e_values(state.e_values)
                
                state = TestState(**{
                    **state.model_dump(),
                    "combined_e_value": combined_e_value,
                    "conclusion": conclusion,
                    "confidence": confidence
                })
            except Exception as e:
                print(f"Error evaluating results: {e}")
                continue
            
            # Check if we've reached a conclusion
            if conclusion is not None:
                break
        
        # If no conclusion was reached, make a final determination
        if state.conclusion is None:
            # Not enough evidence after max iterations
            state = TestState(**{
                **state.model_dump(),
                "conclusion": False,
            })
        
        return state
    
    def _execute_test(self, state: TestState, data: Dict[str, pd.DataFrame]) -> Optional[TestResult]:
        """
        Execute a single test.
        
        Args:
            state: Test state with current proposal
            data: Dictionary of dataframes
            
        Returns:
            Test result or None if execution failed
        """
        try:
            return self.executor_agent.process(state, data)
        except Exception as e:
            print(f"Error executing test: {e}")
            return None