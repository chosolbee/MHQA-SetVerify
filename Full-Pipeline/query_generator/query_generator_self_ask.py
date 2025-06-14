import os
import torch
from vllm import LLM, SamplingParams
from .prompts import SELF_ASK_SYSTEM_PROMPT


class QueryGenerator:
    def __init__(self, llm, max_gen_length=200, temperature=0.7, top_p=0.9):
        os.environ['MKL_THREADING_LAYER']='GNU'

        self.llm = llm
        self.tokenizer = llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_gen_length,
        )

        print("Model loaded successfully.")

    def _gen_retriever_query_prompt(self, question, trace):
        chat = [
            {
                "role": "system",
                "content": SELF_ASK_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question.strip(),
            },
            {
                "role": "assistant",
                "content": trace.strip(),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False,
        )

        return prompt[:-len("<|eot_id|>")] + "\n"

    def batch_generate(self, questions, traces):
        prompts = [self._gen_retriever_query_prompt(question["question"], trace) for question, trace in zip(questions, traces)]

        outputs = self.llm.generate(prompts, self.sampling_params)

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

    questions = ["What county is the city where Peter Kern died in?"]
    traces = ["Follow up: Where did Peter Kern die?\nContext: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries. He is best known as the founder of the confections company that eventually evolved into Kern's Bakery, a brand still marketed in the Knoxville area. The company's former confectionery and ice cream parlor, now called the Mall Building (or Oliver Hotel), still dominates the southwest corner of Market Square. Kern served as Knoxville's mayor from 1890 until 1892. Kern was born in Zwingenberg (near Heidelberg) in Germany\nIntermediate answer: Peter Kern died in Knoxville, Tennessee.\nFollow up: In what county is Knoxville, Tennessee located?\nContext: Knoxville, Tennessee Knoxville is a city in the U.S. state of Tennessee, and the county seat of Knox County. The city had an estimated population of 186,239 in 2016 and a population of 178,874 as of the 2010 census, making it the state's third largest city in the state after Nashville and Memphis. Knoxville is the principal city of the Knoxville Metropolitan Statistical Area, which, in 2016, was 868,546, up 0.9 percent, or 7,377 people, from to 2015. The KMSA is, in turn, the central component of the Knoxville-Sevierville-La Follette Combined Statistical Area, which, in 2013, had a population of"]

    new_traces, responses, is_query_list = query_generator.batch_generate(questions, traces)
    for trace, query, is_query in zip(new_traces, responses, is_query_list):
        print("Trace: ")
        print(trace)
        print(f"Query: {query}")
        print(f"Is Query: {is_query}")
        print("-" * 50)
    print("Test completed.")


if __name__ == "__main__":
    test()
