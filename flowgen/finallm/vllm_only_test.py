import time
from enum import Enum

from pydantic import BaseModel

from flowgen.finallm.vllm_main import vLLM


class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"
    coupe = "Coupe"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

def vllm_structure_output():
    llm = vLLM(
        base_url="http://192.168.170.76:8077/v1",
        model="/home/ng6309/datascience/santhosh/models/Qwen__Qwen3-4B-Thinking-2507"
    )
    st = time.perf_counter()
    response = llm("Generate a JSON with the brand, model and car_type of the most iconic car from the 90's",
                   format=CarDescription)
    et = time.perf_counter()
    print(et-st)
    print(response)

    st = time.perf_counter()
    response = llm(["Generate a JSON with the brand, model and car_type of the most iconic car from the 90's"]*4,
                   format=CarDescription)
    et = time.perf_counter()
    print(et - st)
    print(response)



if __name__ == '__main__':
    #
    vllm_structure_output()
