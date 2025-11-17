import asyncio
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

    async def run(self, messages):
        """Async streaming agent that yields chunks"""
        # Convert string input to message format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Add user messages to state
        self.messages.extend(messages)

        # Stream response from LLM
        full_content = ""
        async for chunk in self.llm.stream(self.messages):
            yield chunk

            # Collect content for state tracking
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    full_content += delta['content']

        # Add complete assistant response to state
        if full_content:
            self.messages.append({
                "role": "assistant",
                "content": full_content
            })


if __name__ == "__main__":
    async def main():
        # Create agent instance
        agent = Agent()

        # Stream response
        async for chunk in agent.run("hi"):
            # Print only content from chunks
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    print(delta['content'], end='', flush=True)

        print()  # New line at the end

    # Run async main
    asyncio.run(main())
