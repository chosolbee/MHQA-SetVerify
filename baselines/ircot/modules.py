import os
import asyncio
from nltk.tokenize import sent_tokenize


class Reasoner:
    def __init__(self, llm, icl_examples, max_gen_length=200, temperature=0.0, top_p=1.0, provider="vllm"):
        os.environ['MKL_THREADING_LAYER']='GNU'

        self.llm = llm
        self.icl_examples = icl_examples

        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p

        self.provider = provider

        print(f"Reasoner - {self.provider} loaded successfully.")

    def _gen_ircot_prompt(self, question, history, trace, fields):
        return self.icl_examples + "\n\n" + \
            "\n\n".join([f"Wikipedia Title: {doc['title']}\n{doc['text']}" for doc in history]) + \
            "\n\n" + "Q: " + question[fields["question"]] + "\n" + "A: " + trace

    def _process_prompts_vllm(self, prompts):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.max_gen_length,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        return [output.outputs[0].text.strip() for output in outputs]

    async def _process_prompts_openai_async(self, prompts):
        from pipeline.modules import AsyncOpenAIProcessor

        async with AsyncOpenAIProcessor(self.llm) as processor:
            return await processor.process_prompts_async(
                prompts,
                max_gen_length=self.max_gen_length,
                temperature=self.temperature,
                top_p=self.top_p,
            )

    def batch_generate_rationales(self, questions, batch_history, traces, fields):
        prompts = [
            self._gen_ircot_prompt(question, history, trace, fields)
            for question, history, trace in zip(questions, batch_history, traces)
        ]

        if self.provider == "vllm":
            rationales = self._process_prompts_vllm(prompts)
        elif self.provider == "openai":
            rationales = asyncio.run(self._process_prompts_openai_async(prompts))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return [sent_tokenize(rationale)[0] if rationale else "" for rationale in rationales]


class QAReader:
    def __init__(self, llm, icl_examples, max_gen_length=200, temperature=0.0, top_p=1.0, provider="vllm"):
        os.environ['MKL_THREADING_LAYER']='GNU'

        self.llm = llm
        self.icl_examples = icl_examples

        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p

        self.provider = provider

        print(f"QA Reader - {self.provider} loaded successfully.")

    def _gen_ircot_prompt(self, question, history, fields):
        return self.icl_examples + "\n\n" + \
            "\n\n".join([f"Wikipedia Title: {doc['title']}\n{doc['text']}" for doc in history]) + \
            "\n\n" + "Q: " + question[fields["question"]] + "\n" + "A: "

    def _process_prompts_vllm(self, prompts):
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.max_gen_length,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        return [output.outputs[0].text.strip() for output in outputs]

    async def _process_prompts_openai_async(self, prompts):
        from pipeline.modules import AsyncOpenAIProcessor

        async with AsyncOpenAIProcessor(self.llm) as processor:
            return await processor.process_prompts_async(
                prompts,
                max_gen_length=self.max_gen_length,
                temperature=self.temperature,
                top_p=self.top_p,
            )

    def _extract_answer(self, rationale):
        if "answer is: " in rationale.lower():
            idx = rationale.lower().find("answer is: ")
            return rationale[idx + len("answer is: "):].split("\n")[0].strip()
        else:
            return rationale

    def batch_generate_answers(self, questions, batch_history, fields):
        prompts = [
            self._gen_ircot_prompt(question, history, fields)
            for question, history in zip(questions, batch_history)
        ]

        if self.provider == "vllm":
            rationales = self._process_prompts_vllm(prompts)
        elif self.provider == "openai":
            rationales = asyncio.run(self._process_prompts_openai_async(prompts))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return [self._extract_answer(rationale) for rationale in rationales]
