import os
import torch
from vllm import LLM, SamplingParams
from .prompts import QUERY_GENERATION_SYSTEM_PROMPT, QUERY_GENERATION_USER_PROMPT


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

        print("Query Generator loaded successfully.")

    def _gen_retriever_query_prompt(self, question, trace):
        chat = [
            {
                "role": "system",
                "content": QUERY_GENERATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "Main question: " + question.strip() + "\n\n" + trace.strip() + "\n\n" + QUERY_GENERATION_USER_PROMPT,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        # ### token수 확인
        # n_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        # print(f"[TOKENS][QG] {n_tokens}")

        return prompt
    
    def extract_query(self, text):
        query = text.strip().splitlines()[0].strip()
        if query.lower().startswith("follow up: ") or query.lower().startswith("follow-up: "):
            query = query[11:].strip()
        return query

    def batch_generate(self, questions: list[str], traces: list[str]) -> tuple[list[str], list[str]]:
        prompts = [
            self._gen_retriever_query_prompt(question["question"], trace)
            for question, trace in zip(questions, traces)
        ]
        outputs = self.llm.generate(prompts, self.sampling_params)

        new_traces = []
        queries = []
        for trace, output in zip(traces, outputs):
            gen_text = output.outputs[0].text
            query = self.extract_query(gen_text)
            new_traces.append(trace + "\nFollow up: " + query.strip())
            queries.append(query)

        return new_traces, queries


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

    questions = [{"question": "What county is the city where Peter Kern died in?"}]
    traces = ["Follow up: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries. He is best known as the founder of the confections company that eventually evolved into Kern's Bakery, a brand still marketed in the Knoxville area. The company's former confectionery and ice cream parlor, now called the Mall Building (or Oliver Hotel), still dominates the southwest corner of Market Square. Kern served as Knoxville's mayor from 1890 until 1892. Kern was born in Zwingenberg (near Heidelberg) in Germany\nIntermediate answer: Peter Kern died in Knoxville, Tennessee."]
    # query should be like -> Follow up: In what county is Knoxville, Tennessee located?


    new_traces, queries = query_generator.batch_generate(questions, traces)
    
    for nt, q in zip(new_traces, queries):
        print("New Trace:", nt)
        print("-" * 50)
        print("Generated Query:", q)
        print("-" * 50)
    print("Test completed.")


if __name__ == "__main__":
    test()
