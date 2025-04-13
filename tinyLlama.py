import torch
import intel_npu_acceleration_library
from intel_npu_acceleration_library.compiler import CompilerConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from intel_npu_acceleration_library.nn.module import NPUModuleWrapper
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaRotaryEmbedding

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def fix_npu_model(model: torch.nn.Module):
    if not isinstance(model, NPUModuleWrapper):
        for _, layer in model.named_children():
            if not hasattr(layer, 'rotary_emb'):
                if isinstance(layer, LlamaAttention):
                    layer.rotary_emb = LlamaRotaryEmbedding
                if isinstance(layer, GemmaAttention):
                    layer.rotary_emb = GemmaRotaryEmbedding
            if not isinstance(layer, NPUModuleWrapper):
                fix_npu_model(layer)

model = AutoModelForCausalLM.from_pretrained(model_path)
fix_npu_model(model)
compiler_conf = CompilerConfig(dtype=torch.bfloat16)
model = intel_npu_acceleration_library.compile(model, compiler_conf)
tokenizer = AutoTokenizer.from_pretrained(model_path)
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