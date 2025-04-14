#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True,  attn_implementation="sdpa")

tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot",
    }
]

while True:
    question = input("Q: ")
    messages.append({"role": "user", "content": question})

    outputs = pipe(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    answer = outputs[0]["generated_text"][-1]
    messages.append(answer)
    print(f"A: {answer['content']}")
