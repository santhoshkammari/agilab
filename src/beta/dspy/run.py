"""
Example: Using the new Agent class
"""
import asyncio
import sys
import aspy as a
sys.path.insert(0, '/home/ntlpt59/master/own/agilab/src/beta')

from transformers.utils import get_json_schema
from aspy.predict.predict import Predict
from aspy.lm.lm import LM

# Configure LM
lm = a.LM(api_base="http://192.168.170.76:8000")
a.configure(lm=lm)


# Define a simple function
def get_weather(location: str, unit: str = "celsius"):
    """Get the weather for a location.

    Args:
        location: The city name
        unit: Temperature unit (celsius or fahrenheit)
    """
    return f"Weather in {location}: 22°{unit[0].upper()}"


def example_tool():
    """Example using Predict with tools"""
    print("=== Tool Example ===")

    # Get schema using transformers.utils
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": get_json_schema(get_weather)
        }
    }]

    # Create predictor with tools
    lm = LM(model="vllm:/home/ng6309/datascience/santhosh/models/Qwen3-14B", api_base="http://192.168.170.76:8000")
    predictor = Predict("question -> answer", lm=lm, tools=tools)

    # Use it
    result = predictor(question="What's the weather in Paris?")
    print(result)
    print()


async def example_simple_call():
    """Simple call - still streams internally"""
    print("=== Simple Call ===")

    agent = a.Agent(system_prompt="You are a helpful math tutor.")

    # Simple call
    result = await agent("What is 15 + 27?")
    print(f"Result: {result}")
    print()


async def example_streaming():
    """Explicit streaming to see events"""
    print("=== Streaming Events ===")

    agent = a.Agent(system_prompt="You are a helpful coding assistant.")

    # Stream events
    async for event in agent.stream("Explain what is async/await in Python in one sentence"):
        print(f"[{event.type}]", end=" ")

        if event.type == "content":
            print(event.content)
        elif event.type == "message_start":
            print("Agent starting...")
        elif event.type == "message_end":
            print(f"Done! Model: {event.metadata.get('model')}")
        elif event.type == "error":
            print(f"Error: {event.content}")
    print()


async def example_conversation():
    """Multi-turn conversation"""
    print("=== Conversation ===")

    agent = a.Agent(system_prompt="You are a friendly assistant.")

    # Turn 1
    response1 = await agent("My name is Alice")
    print(f"Agent: {response1}")

    # Turn 2 - agent remembers
    response2 = await agent("What's my name?")
    print(f"Agent: {response2}")
    print()


async def example_custom_lm():
    """Agent with custom LM (not using global config)"""
    print("=== Custom LM ===")

    custom_lm = a.LM(api_base="http://192.168.170.76:8000")
    agent = a.Agent(
        system_prompt="You are a concise assistant.",
        lm=custom_lm
    )

    result = await agent("Say hello in one word")
    print(f"Result: {result}")
    print()


async def main():
    example_tool()
    await example_simple_call()
    await example_streaming()
    await example_conversation()
    await example_custom_lm()


if __name__ == "__main__":
    asyncio.run(main())


# """
# vllm serve Qwen/Qwen3-4B-Instruct-2507 --gpu-memory-utilization 0.4 --max-model-len 10k
#
#
# #
# # prompt created via XML tagging + dspy signature to prompt style. + followed by output + inputfields.
# # example:
# # <system_prompt>
# # </system_prompt>
# # <output_format>
# # </output_format>
# # <static_input_fields>
# # <static_input_fields>
# # <dynamic_user_level_input_fields>
# # </dynamic_user_level_input_fields>
# # '''
#
#

# import aspy as a
#
# lm = a.LM(api_base="http://192.168.170.76:8000")
# a.configure(lm=lm)
#
# math = a.Predict("q -> think,answer:float")
# res = math(q="2+3?")
# print(type(res),res)


# print("\n=== Testing Predict ===")
# predictor = a.Predict("query -> two_lines")
# result = predictor(query="What is the capital of France?")
# print(result)
#
# print("\n=== Testing Multi-stage Module ===")
#
#
# class DraftArticle(a.Module):
#     def __init__(self):
#         super().__init__()
#         self.build_outline = a.ChainOfThought("topic -> title, two_sections:[str]")
#         self.draft_section = a.ChainOfThought("topic, section_heading -> content")
#
#     def forward(self, topic):
#         outline = self.build_outline(topic=topic)
#         sections = []
#
#         # Handle sections list
#         if hasattr(outline, 'two_sections') and outline.two_sections:
#             for heading in outline.two_sections:
#                 section = self.draft_section(topic=outline.title, section_heading=f"## {heading}")
#                 sections.append(section.content)
#
#         return a.Prediction(title=outline.title, sections=sections)
#
#
# # Zero configuration needed - everything just works!
# draft_article = DraftArticle()
# article = draft_article(topic="World Cup 2002")
# print(article)


# import aspy
#
# # Setup
# lm = aspy.LM(api_base="http://192.168.170.76:8000")
# aspy.configure(lm=lm)
#
# # Create some test examples
# examples = [
#   aspy.Example(question="What is 2+2?", answer="4"),
#   aspy.Example(question="Capital of France?", answer="Paris"),
# ]
#
# # Create a module to evaluate
# math_qa = aspy.Predict("question -> answer")
#
# # Evaluate with progress bar and nice scoring
# evaluator = aspy.Evaluate(
#   devset=examples,
#   metric=aspy.exact_match,
#   display_progress=True,
#   save_as_json="results.json"
# )
#
# result = evaluator(math_qa)
# print(f"Final score: {result.score}%")

#
# import aspy
#
# # Setup
# lm = aspy.LM(api_base="http://192.168.170.76:8000")
# response = lm("""what is 2+3/""")
# content = response['choices'][0]['message']['content']
# print(content)
