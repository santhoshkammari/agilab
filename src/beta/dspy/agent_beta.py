from lm import LM


class Agent:
    def __init__(self, llm: LM = None):
        self.llm = llm or LM(
            model="vllm:/home/ng6309/datascience/santhosh/models/Qwen3-14B",
            api_base="http://localhost:8000"
        )
        self.messages = []

    def __call__(self, messages):
        """Simple agent that forwards messages to the LLM and tracks conversation state"""
        # Convert string input to message format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Add user messages to state
        self.messages.extend(messages)

        # Get response from LLM
        response = self.llm(self.messages)

        # Add assistant response to state
        if response and 'choices' in response and len(response['choices']) > 0:
            assistant_message = response['choices'][0]['message']
            self.messages.append({
                "role": "assistant",
                "content": assistant_message['content']
            })

        return response


if __name__ == "__main__":
    # Create agent instance
    agent = Agent()

    # Send a simple message
    res = agent("hi")

    # Print the result
    print(res)
