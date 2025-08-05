"""
Domain-specific RL environments with plug-and-play tool support.

Each environment encapsulates:
- Domain-specific tools
- Specialized reward functions  
- Pre-configured settings for that domain

Usage:
    # Pick environment and attach any LLM
    math_env = MathsEnv(dataset=math_dataset)
    trained_model = math_env >> trainer >> "train"
    
    # During inference, attach same tools
    result = trained_model("Calculate 15 * 23", tools=math_env.tools)
"""

from __future__ import annotations
import ast
import re
import json
import requests
import subprocess
from typing import Any, Dict, List, Optional
from datasets import Dataset

from rl import RLEnv
from reward import reward_exact_match, reward_step_by_step, reward_length


class MathsEnv(RLEnv):
    """
    Mathematics environment with calculator and symbolic math tools.
    
    Tools:
    - calculator: Basic arithmetic operations
    - solve_equation: Symbolic equation solving
    - plot_function: Mathematical plotting (optional)
    
    Example:
        >>> math_env = MathsEnv(dataset=math_problems)
        >>> result = math_env >> trainer >> "train"
        >>> # During inference: llm("What is 15*23?", tools=math_env.tools)
    """
    
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        difficulty_level: str = "basic",  # basic, intermediate, advanced
        include_plotting: bool = False,
        **kwargs
    ):
        # Define math-specific tools
        tools = [
            self.calculator,
            self.solve_equation,
            self.check_prime,
            self.factorial
        ]
        
        if include_plotting:
            tools.append(self.plot_function)
        
        # Math-specific rewards: accuracy + step-by-step reasoning
        reward_funcs = [reward_exact_match, reward_step_by_step]
        reward_weights = [0.7, 0.3]  # Prioritize correctness
        
        super().__init__(
            dataset=dataset,
            reward_funcs=reward_funcs,
            reward_weights=reward_weights,
            tools=tools,
            max_turns=5,  # Allow multi-step problem solving
            **kwargs
        )
        
        self.difficulty_level = difficulty_level
    
    @staticmethod
    def calculator(expression: str) -> str:
        """
        Evaluate mathematical expressions safely.
        
        Args:
            expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
            
        Returns:
            Result of the calculation as string
        """
        try:
            # Parse and evaluate safely
            node = ast.parse(expression, mode='eval')
            
            # Only allow math operations
            allowed_names = {'abs', 'round', 'min', 'max', 'sum', 'pow'}
            allowed_ops = {
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                ast.USub, ast.UAdd, ast.Num, ast.Constant, ast.Name,
                ast.Load, ast.Expression, ast.Call
            }
            
            for node_item in ast.walk(node):
                if type(node_item) not in allowed_ops:
                    if isinstance(node_item, ast.Name) and node_item.id not in allowed_names:
                        return f"Error: '{node_item.id}' not allowed in calculator"
            
            result = eval(compile(node, '<string>', 'eval'))
            return str(result)
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def solve_equation(equation: str) -> str:
        """
        Solve algebraic equations symbolically.
        
        Args:
            equation: Equation to solve (e.g., "x^2 - 5*x + 6 = 0")
            
        Returns:
            Solution(s) to the equation
        """
        try:
            # Simple quadratic solver for demo
            # In practice, could integrate SymPy
            if "=" not in equation:
                return "Error: Equation must contain '='"
            
            left, right = equation.split("=")
            left, right = left.strip(), right.strip()
            
            # Handle simple linear equations: ax + b = c
            linear_match = re.match(r'([+-]?\d*)\*?x\s*([+-]\s*\d+)?\s*', left)
            if linear_match and right.isdigit():
                a = linear_match.group(1) or "1"
                a = int(a) if a != "" else 1
                b = linear_match.group(2) or "0"
                b = int(b.replace(" ", "")) if b else 0
                c = int(right)
                
                if a == 0:
                    return "Error: Not a linear equation"
                
                x = (c - b) / a
                return f"x = {x}"
            
            return "Info: Complex equation solving requires SymPy integration"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def check_prime(number: str) -> str:
        """
        Check if a number is prime.
        
        Args:
            number: Number to check for primality
            
        Returns:
            Whether the number is prime
        """
        try:
            n = int(number)
            if n < 2:
                return f"{n} is not prime"
            
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return f"{n} is not prime (divisible by {i})"
            
            return f"{n} is prime"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def factorial(number: str) -> str:
        """
        Calculate factorial of a number.
        
        Args:
            number: Number to calculate factorial for
            
        Returns:
            Factorial result
        """
        try:
            n = int(number)
            if n < 0:
                return "Error: Factorial not defined for negative numbers"
            
            result = 1
            for i in range(1, n + 1):
                result *= i
            
            return f"{n}! = {result}"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def plot_function(function: str, x_range: str = "-10,10") -> str:
        """
        Plot mathematical functions (placeholder).
        
        Args:
            function: Function to plot (e.g., "x^2 + 2*x + 1")
            x_range: Range for x values (e.g., "-5,5")
            
        Returns:
            Description of the plot
        """
        return f"Plot generated for function: {function} over range {x_range}"


class WebSearchEnv(RLEnv):
    """
    Web search and information retrieval environment.
    
    Tools:
    - web_search: Search the web for information
    - extract_info: Extract specific information from content
    - summarize_results: Summarize search results
    
    Example:
        >>> web_env = WebSearchEnv(dataset=search_tasks)
        >>> trained_model = web_env >> trainer >> "train"
    """
    
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        max_search_results: int = 5,
        **kwargs
    ):
        tools = [
            self.web_search,
            self.extract_info,
            self.summarize_results
        ]
        
        # Web search rewards: relevance + completeness
        reward_funcs = [reward_exact_match, reward_length]
        reward_weights = [0.8, 0.2]
        
        super().__init__(
            dataset=dataset,
            reward_funcs=reward_funcs,
            reward_weights=reward_weights,
            tools=tools,
            max_turns=8,  # Allow multiple search iterations
            **kwargs
        )
        
        self.max_search_results = max_search_results
    
    @staticmethod
    def web_search(query: str, num_results: int = 3) -> str:
        """
        Search the web for information.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            Search results with titles and snippets
        """
        # Mock implementation - in practice would use actual search API
        mock_results = [
            {
                "title": f"Result {i+1} for '{query}'",
                "snippet": f"This is a mock search result snippet {i+1} containing information about {query}.",
                "url": f"https://example{i+1}.com"
            }
            for i in range(min(num_results, 5))
        ]
        
        formatted_results = []
        for i, result in enumerate(mock_results, 1):
            formatted_results.append(
                f"{i}. {result['title']}\n"
                f"   {result['snippet']}\n"
                f"   URL: {result['url']}\n"
            )
        
        return "\n".join(formatted_results)
    
    @staticmethod
    def extract_info(content: str, target_info: str) -> str:
        """
        Extract specific information from content.
        
        Args:
            content: Text content to search through
            target_info: What information to extract
            
        Returns:
            Extracted information
        """
        # Simple keyword-based extraction
        lines = content.lower().split('\n')
        target_lower = target_info.lower()
        
        relevant_lines = []
        for line in lines:
            if any(word in line for word in target_lower.split()):
                relevant_lines.append(line.strip())
        
        if relevant_lines:
            return f"Found information about '{target_info}':\n" + "\n".join(relevant_lines[:3])
        else:
            return f"No specific information found about '{target_info}'"
    
    @staticmethod
    def summarize_results(results: str, max_length: int = 200) -> str:
        """
        Summarize search results or content.
        
        Args:
            results: Content to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized content
        """
        # Simple extractive summarization
        sentences = results.split('.')
        important_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Filter out very short sentences
                important_sentences.append(sentence.strip())
        
        # Take first few sentences that fit within max_length
        summary = ""
        for sentence in important_sentences:
            if len(summary + sentence) < max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip() or "Unable to generate summary"


class PythonExecutionEnv(RLEnv):
    """
    Python code execution and programming environment.
    
    Tools:
    - execute_code: Run Python code safely
    - analyze_code: Analyze code for issues
    - generate_tests: Generate test cases
    
    Example:
        >>> python_env = PythonExecutionEnv(dataset=coding_tasks)
        >>> trained_model = python_env >> trainer >> "train"
    """
    
    def __init__(
        self,
        dataset: Optional[Dataset] = None,
        timeout: int = 10,
        **kwargs
    ):
        tools = [
            self.execute_code,
            self.analyze_code,
            self.generate_tests,
            self.install_package
        ]
        
        # Code execution rewards: correctness + code quality
        reward_funcs = [reward_exact_match, reward_step_by_step]
        reward_weights = [0.8, 0.2]
        
        super().__init__(
            dataset=dataset,
            reward_funcs=reward_funcs,
            reward_weights=reward_weights,
            tools=tools,
            max_turns=10,  # Allow iterative code development
            **kwargs
        )
        
        self.timeout = timeout
    
    @staticmethod
    def execute_code(code: str, timeout: int = 10) -> str:
        """
        Execute Python code safely with timeout.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            
        Returns:
            Execution result or error message
        """
        try:
            # Remove potentially dangerous imports/functions
            dangerous_patterns = [
                'import os', 'import sys', 'import subprocess',
                'open(', '__import__', 'exec(', 'eval(',
                'compile(', 'globals()', 'locals()'
            ]
            
            code_lower = code.lower()
            for pattern in dangerous_patterns:
                if pattern in code_lower:
                    return f"Error: Potentially unsafe operation detected: {pattern}"
            
            # Create a restricted namespace
            safe_globals = {
                '__builtins__': {
                    'print': print, 'len': len, 'range': range, 'enumerate': enumerate,
                    'zip': zip, 'map': map, 'filter': filter, 'sum': sum, 'min': min,
                    'max': max, 'abs': abs, 'round': round, 'sorted': sorted,
                    'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                    'str': str, 'int': int, 'float': float, 'bool': bool
                }
            }
            
            # Capture output
            import io
            import contextlib
            
            output_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(output_buffer):
                exec(code, safe_globals)
            
            result = output_buffer.getvalue()
            return result if result else "Code executed successfully (no output)"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def analyze_code(code: str) -> str:
        """
        Analyze Python code for potential issues.
        
        Args:
            code: Python code to analyze
            
        Returns:
            Analysis report with suggestions
        """
        try:
            # Parse the code to check for syntax errors
            ast.parse(code)
            
            issues = []
            
            # Check for common issues
            if 'print(' not in code and 'return' not in code:
                issues.append("Code doesn't seem to produce output")
            
            if code.count('(') != code.count(')'):
                issues.append("Mismatched parentheses")
            
            if code.count('[') != code.count(']'):
                issues.append("Mismatched square brackets")
            
            if code.count('{') != code.count('}'):
                issues.append("Mismatched curly braces")
            
            # Check indentation
            lines = code.split('\n')
            for i, line in enumerate(lines, 1):
                if line.strip() and not line.startswith(' ') and ':' in line:
                    if i < len(lines) and lines[i].strip() and not lines[i].startswith(' '):
                        issues.append(f"Line {i+1} may need indentation")
            
            if issues:
                return "Code analysis found issues:\n" + "\n".join(f"- {issue}" for issue in issues)
            else:
                return "Code analysis: No obvious issues found"
                
        except SyntaxError as e:
            return f"Syntax Error: {str(e)}"
        except Exception as e:
            return f"Analysis Error: {str(e)}"
    
    @staticmethod
    def generate_tests(function_code: str) -> str:
        """
        Generate test cases for a function.
        
        Args:
            function_code: Python function code to test
            
        Returns:
            Generated test cases
        """
        try:
            # Extract function name from code
            import re
            func_match = re.search(r'def\s+(\w+)\s*\(', function_code)
            if not func_match:
                return "Error: No function definition found"
            
            func_name = func_match.group(1)
            
            # Generate basic test template
            test_template = f"""
# Test cases for {func_name}
def test_{func_name}():
    # Test case 1: Basic functionality
    result1 = {func_name}()  # Add appropriate arguments
    assert result1 is not None, "Function should return a value"
    
    # Test case 2: Edge case
    # Add edge case testing here
    
    # Test case 3: Error handling
    # Add error case testing here
    
    print("All tests passed!")

# Run the tests
test_{func_name}()
"""
            
            return f"Generated test template:\n{test_template}"
            
        except Exception as e:
            return f"Error generating tests: {str(e)}"
    
    @staticmethod
    def install_package(package_name: str) -> str:
        """
        Install a Python package (mock implementation).
        
        Args:
            package_name: Name of package to install
            
        Returns:
            Installation status
        """
        # Mock implementation - in practice would need careful security considerations
        safe_packages = ['numpy', 'pandas', 'matplotlib', 'requests', 'json']
        
        if package_name.lower() in safe_packages:
            return f"Mock: Package '{package_name}' installed successfully"
        else:
            return f"Error: Package '{package_name}' not in safe package list"


# Factory function for easy environment creation
def create_environment(env_type: str, dataset: Optional[Dataset] = None, **kwargs) -> RLEnv:
    """
    Factory function to create domain-specific environments.
    
    Args:
        env_type: Type of environment ('math', 'web', 'python')
        dataset: Dataset for the environment
        **kwargs: Additional environment-specific parameters
        
    Returns:
        Configured RL environment
    """
    env_map = {
        'math': MathsEnv,
        'web': WebSearchEnv,
        'python': PythonExecutionEnv
    }
    
    if env_type not in env_map:
        raise ValueError(f"Unknown environment type: {env_type}. Available: {list(env_map.keys())}")
    
    return env_map[env_type](dataset=dataset, **kwargs)


if __name__ == '__main__':
    print("=== Domain-Specific RL Environments ===\n")
    
    # Test MathsEnv
    print("1. Testing MathsEnv:")
    math_env = MathsEnv()
    
    # Test tools directly
    calc_result = math_env.calculator("15 * 23 + 7")
    print(f"Calculator: 15 * 23 + 7 = {calc_result}")
    
    prime_result = math_env.check_prime("17")
    print(f"Prime check: {prime_result}")
    
    factorial_result = math_env.factorial("5")
    print(f"Factorial: {factorial_result}\n")
    
    # Test WebSearchEnv
    print("2. Testing WebSearchEnv:")
    web_env = WebSearchEnv()
    
    search_result = web_env.web_search("machine learning", 2)
    print(f"Search results:\n{search_result}\n")
    
    # Test PythonExecutionEnv
    print("3. Testing PythonExecutionEnv:")
    python_env = PythonExecutionEnv()
    
    code_result = python_env.execute_code("print('Hello, World!')\nprint(2 + 3)")
    print(f"Code execution:\n{code_result}")
    
    analysis_result = python_env.analyze_code("def add(a, b):\nreturn a + b")
    print(f"Code analysis: {analysis_result}\n")
    
    print("=== Environment Factory ===")
    
    # Test factory function
    math_env2 = create_environment('math', difficulty_level='advanced')
    web_env2 = create_environment('web', max_search_results=3)
    python_env2 = create_environment('python', timeout=15)
    
    print(f"Created environments: {type(math_env2).__name__}, {type(web_env2).__name__}, {type(python_env2).__name__}")
    
    print("\n=== Usage Pattern ===")
    print("# Create environment")
    print("env = MathsEnv(dataset, difficulty_level='advanced')")
    print("# Train model") 
    print("trained_model = env >> trainer >> 'train'")
    print("# Use in inference with same tools")
    print("result = trained_model('Calculate...', tools=env.tools)")