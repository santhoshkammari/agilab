from flowgen import Gemini, Agent

llm = Gemini()

# Create and use agent
agent = Agent(llm)
result = agent("Calculate 15*23")

print(agent.get_conversation())
exit()

# Later, restore the agent
restored_agent = Agent.load(
    llm=llm,
)

# Continue conversation
continue_result = restored_agent("what was my first question?")
print(continue_result)