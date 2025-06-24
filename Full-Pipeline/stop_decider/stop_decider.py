import os
import torch
from vllm import LLM, SamplingParams
from .prompts import STOP_DECISION_SYSTEM_PROMPT

class StopDecider:
    def __init__(self, llm, max_gen_length=1000, temperature=0.3, top_p=0.9):
        os.environ['MKL_THREADING_LAYER']='GNU'

        self.llm = llm
        self.tokenizer = llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_gen_length,
        )

        print("Stop Decider initialized successfully.")

    def _gen_stop_decision_prompt(self, question, trace):
        chat = [
            {
                "role": "system",
                "content": STOP_DECISION_SYSTEM_PROMPT,
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
            add_generation_prompt=True,
        )
        # ### token수 확인
        # n_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        # print(f"[TOKENS][SD] {n_tokens}")

        return prompt

    def extract_decision(self, text):
        text_upper = text.strip().upper()

        for line in text.strip().splitlines():
            if line.upper().startswith("DECISION:"):
                decision_part = line[9:].strip().upper()
                if "STOP" in decision_part:
                    print("| STOP/CONTINUE: <STOP>")
                    return "STOP"
                elif "CONTINUE" in decision_part:
                    print("| STOP/CONTINUE: <CONTINUE>")
                    return "CONTINUE"

        # Fallback
        if "STOP" in text_upper and "CONTINUE" not in text_upper:
            print("| STOP/CONTINUE: <STOP>")
            return "STOP"
        elif "CONTINUE" in text_upper:
            print("| STOP/CONTINUE: <CONTINUE>") 
            return "CONTINUE"
        else:
            print("| STOP/CONTINUE: <DEFAULT CONTINUE>")
            return "CONTINUE"

    def batch_decide(self, questions, traces):
        prompts = [self._gen_stop_decision_prompt(question["question"], trace) for question, trace in zip(questions, traces)]

        outputs = self.llm.generate(prompts, self.sampling_params) 

        decisions = []
        for output in outputs:
            decision = self.extract_decision(output.outputs[0].text)
            decisions.append(decision)

        return decisions


def test():
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=1,
        quantization=None,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    stop_decider = StopDecider(
        llm=llm,
        max_gen_length=50,
        temperature=0.3,
        top_p=0.9,
    )

    questions = [{"question": "What county is the city where Peter Kern died in?"}]

    # continue
    # traces = ["Query: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries. He is best known as the founder of the confections company that eventually evolved into Kern's Bakery, a brand still marketed in the Knoxville area. The company's former confectionery and ice cream parlor, now called the Mall Building (or Oliver Hotel), still dominates the southwest corner of Market Square. Kern served as Knoxville's mayor from 1890 until 1892. Kern was born in Zwingenberg (near Heidelberg) in Germany\nIntermediate answer: Peter Kern died in Knoxville, Tennessee."]

    # stop
    traces = ["Query: Where did Peter Kern die?\nDocument: Peter Kern (American businessman) Peter Kern (October 31, 1835 – October 28, 1907) was a German-born American businessman and politician active in Knoxville, Tennessee, USA, in the late 19th and early 20th centuries. He is best known as the founder of the confections company that eventually evolved into Kern's Bakery, a brand still marketed in the Knoxville area. The company's former confectionery and ice cream parlor, now called the Mall Building (or Oliver Hotel), still dominates the southwest corner of Market Square. Kern served as Knoxville's mayor from 1890 until 1892. Kern was born in Zwingenberg (near Heidelberg) in Germany.\nIntermediate answer: Peter Kern died in Knoxville, Tennessee.\nQuery: In what county is Knoxville, Tennessee located?\nDocument: Knoxville is a city in the U.S. state of Tennessee, and the county seat of Knox County.\nIntermediate answer: Knoxville is located in Knox County."]

    decisions = stop_decider.batch_decide(questions, traces)
    for question, trace, decision in zip(questions, traces, decisions):
        print(f"Question: {question['question']}")
        print(f"Trace: {trace}")
        print(f"Decision: {decision}")
        print("-" * 50)
    print("Test completed.")


if __name__ == "__main__":
    test()
