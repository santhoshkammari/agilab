from flowgen import Gemini,Agent
from flowgen.llm.basellm import get_weather
def search_web(query):
    return f"Web search results for: {query}"

def get_papers(topic):
    return f"List of papers related to: {topic}"

def analyze_data(data):
    return f"Analysis results for data: {data}"

def create_chart(data):
    return f"Chart created from data: {data}"

llm = Gemini()
# Create specialized agents
research_agent = Agent(llm, tools=[search_web, get_papers])
analysis_agent = Agent(llm, tools=[analyze_data, create_chart])
summary_agent = Agent(llm, tools=[])

# Chain them together
pipeline = research_agent >> analysis_agent >> summary_agent

# Execute the entire pipeline
result = pipeline("Research renewable energy trends and analyze")

print(result['content'])        # Final summary
print(result['chain_length'])   # 3
print(result['chain_results'])  # All intermediate results