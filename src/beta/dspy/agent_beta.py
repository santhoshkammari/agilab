from lm import LM


class Agent:
    def __init__(self, llm: LM = None):
        self.llm = llm or LM(
            model="vllm:/home/ng6309/datascience/santhosh/models/Qwen3-14B",
            api_base="http://localhost:8000"
        )

    def __call__(self, messages):
        """Simple agent that forwards messages to the LLM"""
        response = self.llm(messages)
        return response


if __name__ == "__main__":
    # Create agent instance
    agent = Agent()

    # Send a simple message
    res = agent("hi")

    # Print the result
    print(res)
