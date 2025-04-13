#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from transformers import TextStreamer
import intel_npu_acceleration_library
from intel_npu_acceleration_library.compiler import CompilerConfig
from intel_npu_acceleration_library.nn.module import NPUModuleWrapper
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding
from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaRotaryEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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


model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True,  attn_implementation="sdpa")
fix_npu_model(model)

compiler_conf = CompilerConfig(dtype=intel_npu_acceleration_library.int8)
model = intel_npu_acceleration_library.compile(model, config=compiler_conf).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
streamer = TextStreamer(tokenizer, skip_special_tokens=True)


query = "Hello, who are you?"
prefix = tokenizer(query, return_tensors="pt")["input_ids"]


generation_kwargs = dict(
    input_ids=prefix,
    streamer=streamer,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    max_new_tokens=512,
)

print("Run inference")
_ = model.generate(**generation_kwargs)