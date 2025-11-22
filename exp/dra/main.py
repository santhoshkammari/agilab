import ai

def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for

    Returns:
        A string describing the weather
    """
    weather_data = {
        "london": "Cloudy with a chance of rain, 15°C",
        "paris": "Sunny and pleasant, 18°C",
    }
    city_lower = city.lower()
    return weather_data.get(city_lower, f"Weather data not available for {city}")

lm = ai.LM()
tools = [get_weather]
history = [{"role": "user", "content": "What is the weather in London and Paris? /no_think"}]

async def main():
    result = await ai.agent(
    lm=lm,
    history=history,
    tools=tools,
    max_iterations=5
)
    print(result)
    print(f"Iterations: {result['iterations']}")
    print(f"Total tool calls: {result['tool_calls_total']}")
    print(f"\nFinal Response: {result['final_response'][:200]}...")


import asyncio
asyncio.run(main())

