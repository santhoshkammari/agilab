import sys
sys.path.insert(0, '/home/ntlpt59/master/own/agilab/src/beta')

from transformers.utils import get_json_schema
from aspy.predict.predict import Predict
from aspy.lm.lm import LM

# Define a simple function
def get_weather(location: str, unit: str = "celsius"):
    """Get the weather for a location.

    Args:
        location: The city name
        unit: Temperature unit (celsius or fahrenheit)
    """
    return f"Weather in {location}: 22°{unit[0].upper()}"

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
