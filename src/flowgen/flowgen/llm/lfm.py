from __future__ import annotations
import re
import json
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .basellm import BaseLLM


class LFM(BaseLLM):
    def __init__(self, model, **kwargs):
        super().__init__(model=model, **kwargs)
        
    def _load_llm(self):
        """Load the LFM model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self._model)
        model = AutoModelForCausalLM.from_pretrained(
            self._model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            #attn_implementation="flash_attention_2" # uncomment on compatible GPU
        )
        return model
        
    @staticmethod
    def parse_assistant_response(text):
        pattern = r'<\|im_start\|>assistant(.*?)<\|im_end\|>'
        matches = re.findall(pattern, text, flags=re.DOTALL)

        assistant_response = ""

        if matches:
            assistant_response = matches[-1].strip()

        ## assistant_response : <|tool_call_start|>[web_search(query="definition binomial distribution")]<|tool_call_end|> 
        return assistant_response

    @staticmethod
    def parse_tool_calls(assistant_response: str):
        # Find all tool call blocks
        matches = re.findall(r"<\|tool_call_start\|>\[(.*?)\]<\|tool_call_end\|>", 
                            assistant_response, flags=re.DOTALL)

        tool_calls = []
        for m in matches:
            func_match = re.match(r"(\w+)\((.*)\)", m.strip())
            if not func_match:
                continue

            name, args_str = func_match.groups()
            args = {}

            # Parse key=value pairs
            for k, v1, v2 in re.findall(r'(\w+)\s*=\s*(?:"([^"]*)"|([^,]+))', args_str):
                val = v1 if v1 else v2
                try:
                    val = ast.literal_eval(val)  # convert numbers, booleans, None
                except Exception:
                    pass
                args[k] = val

            tool_calls.append({"name": name, "arguments": args})

        return tool_calls

    def _stream_chat(self, messages, tools, model, **kwargs):
        """Generate streaming text using LFM (not implemented yet)."""
        # For now, just return regular chat response
        # TODO: Implement proper streaming
        return self.chat(messages, tools=tools, model=model, stream=False, **kwargs)
        
    def chat(self, input, **kwargs):
        """Generate text using LFM."""
        input = self._normalize_input(input)
        model = self._check_model(kwargs, self._model)
        
        # Get parameters
        tools = self._get_tools(kwargs)
        timeout = self._get_timeout(kwargs)
        verbose = kwargs.get('verbose', False)
        do_sample = kwargs.get('do_sample', True)
        temperature = kwargs.get('temperature', 0.3)
        min_p = kwargs.get('min_p', 0.15)
        repetition_penalty = kwargs.get('repetition_penalty', 1.05)
        max_new_tokens = kwargs.get('max_new_tokens', 8000)
        
        # Pass tools directly to LFM (no conversion needed)
        
        # Handle streaming
        if kwargs.get("stream"):
            return self._stream_chat(input, tools, model=model, **kwargs)
        
        _prompt = self.tokenizer.apply_chat_template(
            input,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )

        if verbose:
            print('------------')
            print('## INPUT PROMPT ##')
            print(_prompt)
            print('------------')

        input_ids = self.tokenizer.apply_chat_template(
            input,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.llm.device)

        # Filter out parameters that are already set to avoid conflicts
        generation_kwargs = {k: v for k, v in kwargs.items() 
                           if k not in ['do_sample', 'temperature', 'min_p', 'repetition_penalty', 'max_new_tokens', 'tools', 'verbose', 'stream', 'model', 'timeout', 'format']}
        
        output = self.llm.generate(
            input_ids,
            do_sample=do_sample,
            temperature=temperature,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=False)
        
        result = {"think": "", "content": "", "tool_calls": []}
        assistant_response = self.parse_assistant_response(response)
        
        # Extract thinking content from <think> tags
        think, content = self._extract_thinking(assistant_response)
        result['think'] = think
        result['content'] = content
        
        # Parse tool calls
        tool_calls = self.parse_tool_calls(assistant_response)
        if tool_calls:
            for tool_call in tool_calls:
                result['tool_calls'].append({
                    'id': f"call_{hash(str(tool_call))}",
                    'type': 'function',
                    'function': {
                        'name': tool_call['name'],
                        'arguments': json.dumps(tool_call['arguments']) if isinstance(tool_call['arguments'], dict) else str(tool_call['arguments'])
                    }
                })
        
        return result
    


def write_file(filepath: str, content: str) -> None:
    """
    Write the given content to a file.

    Args:
        filepath: The path to the file where the content should be written.
        content: The text content to write into the file.

    Returns:
        None: This function does not return anything.
    """
    pass


def read_file(filepath: str) -> str:
    """
    Read the content of a file.

    Args:
        filepath: The path to the file to be read.

    Returns:
        str: The text content of the file as a string.
    """
    pass


def list_directory(path: str) -> list[str]:
    """
    List all files and directories in the given path.

    Args:
        path: The directory path to list. Defaults to "." (current directory).

    Returns:
        list[str]: A list of file and directory names in the given path.
    """
    pass




if __name__=="__main__":
    checkpoint_path = "/home/ntlpt59/master/models/LiquidAI__LFM2-350M"
    llm = LFM(checkpoint_path)

    message = [{"role":"user","content":"hi"}]
    response = llm(message, tools=[])
    print(response)

    message = [{"role":"user","content":"list files at /home/app "}]
    response = llm(message, tools=[write_file, read_file, list_directory])
    print(response)
