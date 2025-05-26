import os
import torch
from vllm import LLM, SamplingParams
from .prompts import SELF_ASK_SYSTEM_PROMPT, SELF_ASK_USER_PROMPT_FIRST, SELF_ASK_USER_PROMPT_NOT_FIRST


class QueryGenerator:
    def __init__(self, llm, max_gen_length=200, temperature=0.7, top_p=0.9):
        os.environ['MKL_THREADING_LAYER']='GNU'

        self.llm = llm

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_gen_length,
        )

        print("Model loaded successfully.")

    def _gen_retriever_query_prompt(self, trace, is_first=False):
        system_prompt = SELF_ASK_SYSTEM_PROMPT
        user_prompt = trace + (SELF_ASK_USER_PROMPT_FIRST if is_first else SELF_ASK_USER_PROMPT_NOT_FIRST)
        prompt = [
            {
                "role": "system",
                "content":  system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ]
        return prompt

    def batch_generate(self, traces, is_first=False):
        prompts = [self._gen_retriever_query_prompt(trace, is_first) for trace in traces]

        outputs = self.llm.chat(prompts, self.sampling_params)

        new_traces = []
        responses = []
        is_query_list = []
        for trace, output in zip(traces, outputs):
            new_trace, response, is_query = self.extract_query(output.outputs[0].text)
            trace += new_trace
            new_traces.append(trace)
            responses.append(response)
            is_query_list.append(is_query)

        return new_traces, responses, is_query_list

    def extract_query(self, text):
        lines = text.strip().split('\n')
        trace = ""
        for line in lines:
            line_text = line.strip()
            if line_text.lower().startswith("so the final answer is: "):
                return trace, line_text[len("so the final answer is: "):].strip(), False
            trace += line_text + "\n"
            if line_text.lower().startswith("follow up: ") or line_text.lower().startswith("follow-up: "):
                return trace, line_text[len("follow up: "):].strip(), True
        return trace, "", True


def test():
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        quantization=None,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    query_generator = QueryGenerator(
        llm=llm,
        max_gen_length=2048,
        temperature=0.7,
        top_p=0.9,
    )

    traces = [(
        "Question: Country A has an embassy from the country that produces the show Krystala. Who was a prominent figure in the radio division of the network that created a version of the Biggest Loser for country A?\n"
        "Are follow up questions needed here: Yes.\n"
        "Follow up: Which country produces the show Krystala?\n"
        "Context: Krystala is a daily fantasy/sci-fi/adventure/soap opera serial (superserye/fantaserye) from the Philippines, where it was produced by and aired on ABS-CBN from October 11, 2004 to April 22, 2005. The show also aired simultaneously on The Filipino Channel and on a one-week delay on International Channel (now AZN-TV) in the United States.\n"
    )]
    is_first = False

    new_traces, responses, is_query_list = query_generator.batch_generate(traces, is_first)
    for trace, query, is_query in zip(new_traces, responses, is_query_list):
        print("Trace: ")
        print(trace)
        print(f"Query: {query}")
        print(f"Is Query: {is_query}")
        print("-" * 50)
    print("Test completed.")


if __name__ == "__main__":
    test()
