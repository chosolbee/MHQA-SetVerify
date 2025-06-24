import asyncio
from dataclasses import dataclass
import httpx
from openai import AsyncOpenAI


@dataclass
class OpenAIConfig:
    model_id: str
    max_retries: int = 1
    batch_timeout: int = 600  # seconds
    total_timeout: int = 60  # seconds
    connect_timeout: int = 10  # seconds
    max_keepalive_connections: int = 10
    max_connections: int = 20
    max_concurrent: int = 3
    api_key: str = None  # Optional


class AsyncOpenAIProcessor:
    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = None

    async def __aenter__(self):
        timeout = httpx.Timeout(self.config.total_timeout, connect=self.config.connect_timeout)
        limits = httpx.Limits(max_keepalive_connections=self.config.max_keepalive_connections, max_connections=self.config.max_connections)

        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            timeout=self.config.total_timeout,
            max_retries=self.config.max_retries,
            http_client=httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
            ),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.close()

    async def _process_prompt_async(self, index, prompt, max_gen_length, temperature, top_p, semaphore):
        async with semaphore:
            try:
                response = await self.client.responses.create(
                    model=self.config.model_id,
                    input=prompt,
                    max_output_tokens=max_gen_length,
                    temperature=temperature,
                    top_p=top_p,
                )
                return {
                    "index": index,
                    "content": response.output[0].content[0].text,
                }
            except Exception as e:
                print(f"[OpenAI] Error processing prompt: {e}")
                return {"index": index, "content": "error"}

    async def process_prompts_async(self, prompts, max_gen_length, temperature, top_p):
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        tasks = []
        for i, prompt in enumerate(prompts):
            task = self._process_prompt_async(i, prompt, max_gen_length, temperature, top_p, semaphore)
            tasks.append(task)

        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.batch_timeout,
            )

            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({'index': i, 'content': "Error: " + str(result)})
                else:
                    processed_results.append(result)

            processed_results = sorted(processed_results, key=lambda x: x['index'])

            return [result["content"] for result in processed_results]
        except asyncio.TimeoutError:
            print("[OpenAI] Batch processing timed out")
            return ["Error: Batch Timeout" for i in range(len(prompts))]
