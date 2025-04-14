# Run the TinyLlama model on the NPU
#### Intel NPU
The Intel NPU is an AI accelerator integrated into the Intel Core Ultra processors, which features a unique architecture that includes computing acceleration and data transmission capabilities.
#### Llama
Llama (Large Language Model Meta AI, formerly known as LLaMA) is a series of Large Language Models (LLMs) released by Meta AI starting in February 2023.
A Large Language Model (LLM) is a machine learning model designed for natural language processing tasks such as language generation. LLMs are language models with many parameters, trained through self-supervised learning on a large set of text.

## Setup

Check that your system has an available NPU ([how-to](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html)).

Before starting
- 1.Install Python.
- 2.Open PowerShell in your home directory. For example, C:/Users/raymond.
- 3.Run the command mkdir tmpbuild This creates a new folder called tmpbuild in your home directory, which will have a much shorter path than the default pip install version.
- 4.Run the command $env:TMPDIR="tmpbuild" This sets the build directory for pip install so that it will use the folder that you just created. Don't worry, this won't permanently change your system environment variables. Your changes will be reset once you close your PowerShell window.

Finally, run the command

```bash
   pip install intel-npu-acceleration-library
```
To build the package you need a compiler in your system (Visual Studio, Rust and CMake suggested for Windows build). MacOS is not yet supported. At the moment only Ubuntu OS is supported for Linux build.

#### After the installation of intel-npu-acceleration-library
The code does not work on the intel_npu_acceleration_library 1.4.0, until change the ‘intel_npu_acceleration_library/nn/llm.py‘’’ line 245, from
```
return attn_output, None, past_key_value
```
to
```
return attn_output, None
```
#### Use the model from vLLM
```
# Install vLLM from pip:
pip install vllm
```

## Usage

#### tinyLlamaChat.py
It allows you to run a Q&A session.

#### tinyLlamaDemo.py
It allows you to automatically generate question-answer examples.
