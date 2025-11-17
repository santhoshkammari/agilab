import asyncio
from lm import LM
from transformers.utils import get_json_schema


class Agent:
    def __init__(self, llm: LM = None):
        self.llm = llm or LM(
            model="vllm:/home/ng6309/datascience/santhosh/models/Qwen3-14B",
            api_base="http://localhost:8000"
        )
        self.messages = []
        self.tools = {}

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

    async def run(self, messages, tools=None):
        """Async streaming agent that yields chunks"""
        # Add user messages to state
        self.messages.extend(messages)

        # Register tools if provided
        if tools:
            for tool in tools:
                self.tools[tool.__name__] = tool

        # Convert tools to schemas on-the-fly
        tool_schemas = [get_json_schema(tool) for tool in tools] if tools else None

        # Stream response from LLM
        full_content = ""
        async for chunk in self.llm.stream(self.messages, tools=tool_schemas):
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
    # Define a sample tool
    def get_weather(city: str, unit: str = "celsius"):
        """
        Get the current weather for a city

        Args:
            city: The name of the city
            unit: Temperature unit (choices: ["celsius", "fahrenheit"])
        """
        return f"Weather in {city}: 22 degrees {unit}"

    async def main():
        # Create agent instance
        agent = Agent()

        # Stream response with tools
        messages = [{"role": "user", "content": "What's the weather in Paris?"}]
        async for chunk in agent.run(messages, tools=[get_weather]):
            # Print only content from chunks
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    print(delta['content'], end='', flush=True)
                # Also print tool calls if present
                if 'tool_calls' in delta:
                    print(f"\nTool call: {delta['tool_calls']}")

        print()  # New line at the end

    # Run async main
    asyncio.run(main())
