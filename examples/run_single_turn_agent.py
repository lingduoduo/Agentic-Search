import asyncio

from transformers import AutoTokenizer

from src.agent_loop import SingleTurnAgentLoop, SingleTurnAgentLoopConfig
from examples.run_agentic_search import LocalServerManager

model = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
server_manager = LocalServerManager(
    model_path=model,
    device="cpu",
    generation_timeout_seconds=45,
)
loop = SingleTurnAgentLoop(
    tokenizer=tokenizer,
    server_manager=server_manager,
    config=SingleTurnAgentLoopConfig(
        force_search=True,
        search_url="http://127.0.0.1:8000/retrieve",
        topk=2,
        response_length=128,
    ),
)


async def main():
    output = await loop.run(
        [{"role": "user", "content": "What is FAISS?"}],
        {"temperature": 0, "max_tokens": 128},
    )
    print(output.final_answer)


asyncio.run(main())
