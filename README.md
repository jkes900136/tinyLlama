# TinyLlama on the Intel NPU
The Intel NPU Acceleration Library is a Python library designed to leverage the power of the Intel Neural Processing Unit (NPU) to perform high-speed computations on Intel Core Ultra and later hardware, thereby improving the efficiency of applications.

## Setup

Check that your system has an available NPU ([how-to](https://www.intel.com/content/www/us/en/support/articles/000097597/processors.html)).

Before starting
- 1.Install Python
- 2.Open PowerShell in your home directory. For example, C:/Users/raymond
- 3.Run the command mkdir tmpbuild This creates a new folder called tmpbuild in your home directory, which will have a much shorter path than the default pip install version
- 4.Run the command $env:TMPDIR="tmpbuild" This sets the build directory for pip install so that it will use the folder that you just created. Don't worry, this won't permanently change your system environment variables; your changes will be reset once you close your PowerShell window

Finally, run the command

```bash
   pip install intel-npu-acceleration-library
```
To build the package you need a compiler in your system (Visual Studio suggested for Windows build). MacOS is not yet supported. At the moment only Ubuntu OS is supported for Linux build.

#### After the installation of intel-npu-acceleration-library
the code does not work on the intel_npu_acceleration_library 1.4.0, until change the ‘intel_npu_acceleration_library/nn/llm.py‘’’ line 245, from
```
return attn_output, None, past_key_value
```
to
```
return attn_output, None
```
## Usage

#### tinyLlama.py
It allows you to run a Q&A session.

#### tLlamaSample.py
It allows you to automatically generate question-answer examples.
