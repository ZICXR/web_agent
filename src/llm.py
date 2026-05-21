import math
from openai import OpenAI
from utils.load_env import load_env
from utils.calculate import calculate_entropy, sliding_window_stats

base_url = load_env("OPENAI_BASE_URL")
api_key = load_env("OPENAI_API_KEY")
model = load_env("OPENAI_MODEL")


class LLM():
    def __init__(self,):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def invoke(self, prompt: str, max_tokens: int=512, is_loggrobs: bool=True, top_logprobs: int=5, temperature=0):
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            logprobs=is_loggrobs,
            top_logprobs=top_logprobs if is_loggrobs else None,
            temperature=temperature
        )
        res = {}
        if is_loggrobs:
            res = calculate_entropy(response)
        else:
            content = response.choices[0].message.content
            res = {"content": content, "results": {}}

        return res

    def invoke_entropy(self, prompt: str, max_tokens: int=512, top_logprobs: int=5, temperature=0):
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
            temperature=temperature
        )
        res = calculate_entropy(response)
        results = sliding_window_stats(res["results"])
        ans = {"window": results, "data": res}
        return ans







