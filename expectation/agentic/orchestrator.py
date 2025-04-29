from typing import Dict
import pandas as pd
from expectation.agentic.models import TestState, AgentConfig
from expectation.agentic.agents import DesignerAgent, ExecutionAgent, EvaluationAgent

class HypothesisTestOrchestrator:
    """
    Orchestrates the sequential falsification testing workflow.
    
    This class manages the state transitions and coordinates the
    overall testing process.
    """
    
    def __init__(
        self, 
        designer_agent: DesignerAgent,
        executor_agent: ExecutionAgent,
        evaluator_agent: EvaluationAgent,
        config: AgentConfig
    ):
        self.designer_agent = designer_agent
        self.executor_agent = executor_agent
        self.evaluator_agent = evaluator_agent
        self.config = config
    
    def test_hypothesis(
        self, 
        hypothesis: str, 
        data: Dict[str, pd.DataFrame]
    ) -> TestState:


        state = TestState(hypothesis=hypothesis)
        

        for iteration in range(1, self.config.max_iterations + 1):

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

                combined_e_value = 1.0
                if state.e_values:
                    if self.config.combine_method == "product":
                        combined_e_value = pd.Series(state.e_values).prod()
                
                state = TestState(**{
                    **state.model_dump(),
                    "combined_e_value": combined_e_value,
                    "conclusion": conclusion,
                    "confidence": confidence
                })
            except Exception as e:
                print(f"Error evaluating results: {e}")
                continue
            
            if conclusion is not None:
                break
        
        if state.conclusion is None:
            # Not enough evidence after max iterations
            state = TestState(**{
                **state.model_dump(),
                "conclusion": False,
            })
        
        return state