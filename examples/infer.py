from flowgen import Gemini, Agent

llm = Gemini()

# Create and use agent
agent = Agent(llm)
result = agent("Calculate 15*23",)


# Continue conversation
continue_result = restored_agent("what was my first question?")
print(continue_result)