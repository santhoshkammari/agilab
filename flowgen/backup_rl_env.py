from __future__ import annotations
import json
import inspect
from typing import Any, Callable, Optional, Union, List, Dict
from datasets import Dataset
from collections import defaultdict


class RLEnv:
    """
    Universal RL Environment for LLM training and evaluation.
    
    Designed with the same philosophy as the LLM framework:
    - Simple, pythonic interface
    - Auto-detection of usage patterns
    - Seamless integration with HuggingFace datasets
    - Loose coupling with any trainer (GRPOTrainer, PPOTrainer, etc.)
    
    Example:
        >>> env = RLEnv(dataset=dataset, reward_funcs=[accuracy_reward], reward_weights=[1.0])
        >>> results = env(llm)  # Auto-evaluate on dataset
        >>> result = env(llm, "What is 2+2?")  # Single rollout
        >>> trainer = env.to_grpo_trainer(model)  # Direct trainer creation
    """
    
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        reward_funcs: Optional[Union[Callable, List[Callable]]] = None,
        reward_weights: Optional[List[float]] = None,
        tools: Optional[List[Callable]] = None,
        max_turns: int = 1,
        **kwargs
    ):
        """
        Initialize RL Environment.
        
        Args:
            dataset: HuggingFace dataset with 'prompt' column (required for trainers)
            reward_funcs: Single reward function or list of reward functions
            reward_weights: Weights for combining multiple rewards (defaults to equal weights)
            tools: List of tool functions for multi-turn interactions
            max_turns: Maximum conversation turns (1 for single-turn tasks)
            **kwargs: Additional parameters passed to rollouts
        """
        self.dataset = dataset
        
        # Normalize reward functions to list
        if reward_funcs is None:
            self.reward_funcs = []
        elif callable(reward_funcs):
            self.reward_funcs = [reward_funcs]
        else:
            self.reward_funcs = list(reward_funcs)
        
        # Set default weights
        if reward_weights is None:
            self.reward_weights = [1.0] * len(self.reward_funcs)
        else:
            self.reward_weights = reward_weights
            
        if len(self.reward_weights) != len(self.reward_funcs):
            raise ValueError(f"Number of reward_weights ({len(self.reward_weights)}) must match number of reward_funcs ({len(self.reward_funcs)})")
        
        self.tools = tools or []
        self.max_turns = max_turns
        self.kwargs = kwargs
    
    def __call__(self, llm, input=None, **kwargs) -> Union[Dict, List[Dict]]:
        """
        Universal interface - auto-detects usage pattern.
        
        Args:
            llm: LLM instance with __call__ method
            input: None (dataset eval), string (single rollout), or list (batch rollouts)
            **kwargs: Additional parameters for generation
            
        Returns:
            Dictionary with results or list of result dictionaries
        """
        if input is None and self.dataset:
            # Evaluate on entire dataset
            return self.evaluate(llm, **kwargs)
        elif isinstance(input, str):
            # Single rollout
            return self.rollout(llm, input, **kwargs)
        elif isinstance(input, list):
            # Batch rollouts
            return [self.rollout(llm, x, **kwargs) for x in input]
        else:
            raise ValueError("Input must be None (dataset eval), string, or list of strings")
    
    def rollout(self, llm, prompt: str, answer: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Single environment interaction.
        
        Args:
            llm: LLM instance
            prompt: Input prompt
            answer: Ground truth answer (optional, used for reward calculation)
            **kwargs: Additional parameters for LLM generation
            
        Returns:
            Dictionary with messages, reward, turns, and metadata
        """
        messages = [{"role": "user", "content": prompt}]
        
        for turn in range(self.max_turns):
            # Generate response
            response = llm(messages, tools=self.tools, **{**self.kwargs, **kwargs})
            
            # Add assistant response
            messages.append({
                "role": "assistant", 
                "content": response.get('content', '')
            })
            
            # Handle tool calls
            if response.get('tools'):
                for tool_call in response['tools']:
                    result = self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "name": tool_call['name'],
                        "content": str(result)
                    })
                continue
            
            # Single turn or natural completion
            break
        
        # Calculate reward
        completion = response.get('content', '')
        total_reward = self._compute_reward(prompt, completion, answer, **kwargs)
        
        return {
            "messages": messages,
            "completion": completion,
            "reward": total_reward,
            "turns": turn + 1,
            "prompt": prompt,
            "answer": answer
        }
    
    def evaluate(self, llm, num_examples: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Evaluate LLM on the dataset.
        
        Args:
            llm: LLM instance
            num_examples: Number of examples to evaluate (None for all)
            **kwargs: Additional parameters for generation
            
        Returns:
            Dictionary with evaluation results and statistics
        """
        if self.dataset is None:
            raise ValueError("Dataset must be provided for evaluation")
        
        # Sample dataset if requested
        eval_dataset = self.dataset
        if num_examples is not None:
            eval_dataset = eval_dataset.select(range(min(num_examples, len(eval_dataset))))
        
        results = []
        total_reward = 0.0
        
        for example in eval_dataset:
            prompt = example['prompt']
            answer = example.get('answer', None)
            
            # Add any additional dataset columns to kwargs
            example_kwargs = {k: v for k, v in example.items() if k not in ['prompt', 'answer']}
            
            result = self.rollout(llm, prompt, answer, **{**example_kwargs, **kwargs})
            results.append(result)
            total_reward += result['reward']
        
        return {
            "results": results,
            "mean_reward": total_reward / len(results) if results else 0.0,
            "num_examples": len(results),
            "rewards": [r['reward'] for r in results]
        }
    
    def _execute_tool(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a tool function call."""
        tool_name = tool_call['name']
        tool_args = tool_call.get('arguments', {})
        
        # Find the tool function
        tool_func = None
        for tool in self.tools:
            if tool.__name__ == tool_name:
                tool_func = tool
                break
        
        if tool_func is None:
            return f"Error: Tool {tool_name} not found"
        
        try:
            return tool_func(**tool_args)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def _compute_reward(self, prompt: str, completion: str, answer: Optional[str] = None, **kwargs) -> float:
        """Compute weighted sum of all reward functions."""
        if not self.reward_funcs:
            return 0.0
        
        total_reward = 0.0
        for reward_func, weight in zip(self.reward_funcs, self.reward_weights):
            try:
                # Check function signature to pass appropriate arguments
                sig = inspect.signature(reward_func)
                func_kwargs = {}
                
                # Add available arguments that the function accepts
                if 'prompt' in sig.parameters:
                    func_kwargs['prompt'] = prompt
                if 'completion' in sig.parameters:
                    func_kwargs['completion'] = completion
                if 'answer' in sig.parameters and answer is not None:
                    func_kwargs['answer'] = answer
                
                # Add any additional kwargs the function accepts
                for param_name in sig.parameters:
                    if param_name in kwargs:
                        func_kwargs[param_name] = kwargs[param_name]
                
                # If function accepts **kwargs, pass everything
                if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                    func_kwargs.update(kwargs)
                
                reward = reward_func(**func_kwargs)
                total_reward += reward * weight
                
            except Exception as e:
                print(f"Warning: Reward function {reward_func.__name__} failed: {e}")
                continue
        
        return total_reward
    
    # GRPOTrainer Integration
    def get_grpo_components(self) -> Dict[str, Any]:
        """Returns dataset and reward_funcs compatible with GRPOTrainer."""
        if self.dataset is None:
            raise ValueError("Dataset is required for GRPOTrainer")
        
        if 'prompt' not in self.dataset.column_names:
            raise ValueError("Dataset must contain 'prompt' column for GRPOTrainer")
        
        return {
            'train_dataset': self.dataset,
            'reward_funcs': self._make_grpo_reward_funcs(),
            'reward_weights': self.reward_weights if len(self.reward_funcs) > 1 else None
        }
    
    def _make_grpo_reward_funcs(self) -> List[Callable]:
        """Convert our reward functions to GRPOTrainer format."""
        if not self.reward_funcs:
            return []
        
        grpo_funcs = []
        for reward_func in self.reward_funcs:
            def make_grpo_wrapper(func):
                def grpo_wrapper(prompts, completions, completions_ids, trainer_state=None, **kwargs):
                    """GRPOTrainer compatible reward function."""
                    rewards = []
                    
                    for prompt, completion in zip(prompts, completions):
                        try:
                            # Check if our function accepts trainer_state
                            sig = inspect.signature(func)
                            func_kwargs = {}
                            
                            if 'prompt' in sig.parameters:
                                func_kwargs['prompt'] = prompt
                            if 'completion' in sig.parameters:
                                func_kwargs['completion'] = completion
                            if 'trainer_state' in sig.parameters:
                                func_kwargs['trainer_state'] = trainer_state
                            
                            # Add dataset columns
                            for param_name in sig.parameters:
                                if param_name in kwargs:
                                    func_kwargs[param_name] = kwargs[param_name]
                            
                            # If function accepts **kwargs, pass everything
                            if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                                func_kwargs.update(kwargs)
                            
                            reward = func(**func_kwargs)
                            rewards.append(float(reward))
                            
                        except Exception as e:
                            print(f"Warning: Reward function {func.__name__} failed for prompt: {e}")
                            rewards.append(0.0)
                    
                    return rewards
                
                return grpo_wrapper
            
            grpo_funcs.append(make_grpo_wrapper(reward_func))
        
        return grpo_funcs
    
    def to_grpo_trainer(self, model, **trainer_kwargs):
        """Create GRPOTrainer directly."""
        try:
            from trl import GRPOTrainer, GRPOConfig
        except ImportError:
            raise ImportError("TRL library is required for GRPOTrainer. Install with: pip install trl")
        
        components = self.get_grpo_components()
        
        # Create default config if not provided
        args = trainer_kwargs.pop('args', None)
        if args is None:
            config_kwargs = {k: v for k, v in trainer_kwargs.items() 
                           if k in GRPOConfig.__dataclass_fields__}
            other_kwargs = {k: v for k, v in trainer_kwargs.items() 
                          if k not in GRPOConfig.__dataclass_fields__}
            
            args = GRPOConfig(
                output_dir=config_kwargs.pop('output_dir', './grpo_output'),
                **config_kwargs
            )
            trainer_kwargs = other_kwargs
        
        return GRPOTrainer(
            model=model,
            train_dataset=components['train_dataset'],
            reward_funcs=components['reward_funcs'],
            args=args,
            **trainer_kwargs
        )
    
    # Generic Trainer Support
    def for_trainer(self, trainer_type: str = "grpo") -> Dict[str, Any]:
        """Generate trainer-specific components."""
        trainer_type = trainer_type.lower()
        
        if trainer_type == "grpo":
            return self.get_grpo_components()
        elif trainer_type == "ppo":
            return self.get_ppo_components()
        elif trainer_type == "dpo":
            return self.get_dpo_components()
        else:
            # Generic format
            return {
                'dataset': self.dataset,
                'reward_functions': self.reward_funcs,
                'weights': self.reward_weights,
                'tools': self.tools
            }
    
    def get_ppo_components(self) -> Dict[str, Any]:
        """PPOTrainer compatibility."""
        return {
            'dataset': self.dataset,
            'reward_model': self._create_reward_model(),
        }
    
    def get_dpo_components(self) -> Dict[str, Any]:
        """DPOTrainer compatibility."""
        return {
            'train_dataset': self._create_preference_dataset(),
        }
    
    def _create_reward_model(self):
        """Create a reward model wrapper for PPO-style training."""
        class RewardModel:
            def __init__(self, env):
                self.env = env
            
            def __call__(self, prompts, completions):
                rewards = []
                for prompt, completion in zip(prompts, completions):
                    reward = self.env._compute_reward(prompt, completion)
                    rewards.append(reward)
                return rewards
        
        return RewardModel(self)
    
    def _create_preference_dataset(self):
        """Create preference pairs for DPO training."""
        if self.dataset is None:
            raise ValueError("Dataset required for preference generation")
        
        # This would generate preference pairs by comparing rewards
        # Implementation depends on specific DPO requirements
        raise NotImplementedError("DPO preference dataset generation not yet implemented")


# Common reward functions
def exact_match_reward(prompt: str, completion: str, answer: str, **kwargs) -> float:
    """Reward exact matches with ground truth."""
    return 1.0 if answer.strip().lower() in completion.lower() else 0.0


def length_reward(prompt: str, completion: str, target_length: int = 100, **kwargs) -> float:
    """Reward completions close to target length."""
    return max(0.0, 1.0 - abs(len(completion) - target_length) / target_length)


def step_by_step_reward(prompt: str, completion: str, **kwargs) -> float:
    """Reward step-by-step reasoning (more line breaks = better)."""
    steps = completion.count('\n') + completion.count('. ')
    return min(1.0, steps / 5.0)


def format_reward(prompt: str, completion: str, required_format: str = None, **kwargs) -> float:
    """Reward specific formatting patterns."""
    if required_format is None:
        return 1.0
    
    import re
    pattern = required_format
    return 1.0 if re.search(pattern, completion) else 0.0


# Example usage and testing
if __name__ == '__main__':
    print("=== RL Environment Examples ===\n")
    
    # Create sample dataset
    from datasets import Dataset
    
    sample_data = Dataset.from_list([
        {"prompt": "What is 2+3?", "answer": "5"},
        {"prompt": "What is 7*8?", "answer": "56"},
        {"prompt": "What is 10-4?", "answer": "6"},
    ])
    
    # Example 1: Simple exact match environment
    print("1. Simple exact match environment:")
    env = RLEnv(
        dataset=sample_data,
        reward_funcs=exact_match_reward,
        max_turns=1
    )
    
    # Mock LLM for testing
    class MockLLM:
        def __call__(self, messages, **kwargs):
            last_msg = messages[-1]['content']
            if "2+3" in last_msg:
                return {"content": "The answer is 5"}
            elif "7*8" in last_msg:
                return {"content": "7 times 8 equals 56"}
            else:
                return {"content": "I don't know"}
    
    mock_llm = MockLLM()
    result = env(mock_llm, "What is 2+3?", answer="5")
    print(f"Single rollout reward: {result['reward']}")
    print(f"Completion: {result['completion']}\n")
    
    # Example 2: Multiple rewards with weights
    print("2. Multiple rewards with weights:")
    multi_env = RLEnv(
        dataset=sample_data,
        reward_funcs=[exact_match_reward, step_by_step_reward, length_reward],
        reward_weights=[1.0, 0.3, 0.1],  # Accuracy most important
        max_turns=1
    )
    
    result = multi_env(mock_llm, "What is 2+3?", answer="5", target_length=20)
    print(f"Multi-reward result: {result['reward']}\n")
    
    # Example 3: GRPOTrainer components
    print("3. GRPOTrainer compatibility:")
    try:
        components = env.get_grpo_components()
        print(f"Dataset columns: {components['train_dataset'].column_names}")
        print(f"Number of reward functions: {len(components['reward_funcs'])}")
        print("✓ GRPOTrainer compatible\n")
    except Exception as e:
        print(f"GRPOTrainer error: {e}\n")
    
    # Example 4: Evaluation on dataset
    print("4. Dataset evaluation:")
    try:
        eval_results = env(mock_llm, num_examples=2)
        print(f"Mean reward: {eval_results['mean_reward']:.2f}")
        print(f"Individual rewards: {eval_results['rewards']}")
    except Exception as e:
        print(f"Evaluation error: {e}")
    
    print("\n=== Usage Patterns ===")
    print("• Single rollout: env(llm, 'prompt')")
    print("• Dataset eval: env(llm)")
    print("• Batch process: env(llm, ['prompt1', 'prompt2'])")
    print("• GRPOTrainer: env.to_grpo_trainer(model)")
    print("• Generic trainer: env.for_trainer('ppo')")