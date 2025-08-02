# from datasets import Dataset
# from llm import vLLM
#
# base_llm = vLLM(
#     host='192.168.170.76',
#     port="8077",
#     model="/home/ng6309/datascience/santhosh/models/Qwen3-14B"
# )
#
# def extract_number(batch):
#     prompts = [f"Extract number and return it: {text}" for text in batch['input']]
#     results = base_llm(prompts)
#     return {'output': [r['content'] for r in results]}
#
#
# if __name__ == '__main__':
#     queries = ["I have 25 apples", "There are 100 students", "Cost is $45.99", "Age is 30 years"]
#     dataset = Dataset.from_dict({'input': queries}).map(extract_number, batch_size=4, batched=True)
#     print(dataset['output'])
#

