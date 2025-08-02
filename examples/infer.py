from flowgen import Gemini, Agent

llm = Gemini()

# Create and use agent
agent = Agent(llm)
result = agent("Calculate 15*23",)