# Welcome to Tuningtron!

![tuningtron_left](https://github.com/user-attachments/assets/4de3a715-36cc-4ee9-b823-9920ef5785f8)

**Tuningtron** is a library built on top of **Hugging Face Transformers**, designed to simplify the process of fine-tuning large language models (LLMs) for developers. It focuses on making LLM fine-tuning feasible even with limited computational resources, such as Nvidia GeForce RTX 3090 GPUs. The library supports training on both GPUs and CPUs, and it includes a feature for offloading model weights to the CPU when using a single GPU. With one **Nvidia GeForce RTX 3090 GPU** and **256 GB of RAM**, Tuningtron can handle fine-tuning models with up to approximately **70 billion** parameters.

## Environment
The Tuningtron library is compatible with the Ubuntu 22.04 operating system. To set up the required environment for this library, system tools must be installed using the command:
```console
sudo apt install -y python3-pip ccache make cmake g++ mpich
```
To create a Python virtual environment with a GPU, use the command:
```console
conda env create -f environment.yml
```
In the absence of a GPU, the environment can be set up with the command: 
```console
conda env create -f environment-cpu.yml
```
These steps ensure that all necessary dependencies are correctly configured, allowing the Tuningtron library to function optimally.

## Installation
```console
pip install tuningtron
```

## Updating operating system drivers
The following commands allow you to update operating system drivers:
```console
sudo rm -r /var/lib/dkms/nvidia
sudo dpkg -P --force-all $(dpkg -l | grep "nvidia-" | grep -v lib | awk '{print $2}')
sudo ubuntu-drivers install
```

## Using swap
When fine-tuning models with a large number of parameters, it might be necessary to increase the operating system's swap space. This can be done using the following steps:

```console
sudo swapoff -a
sudo fallocate -l 50G
sudo chmod 600
sudo mkswap /swapfile
sudo swapon /swapfile
```

These commands will increase the swap space, providing additional virtual memory that can help manage the large memory requirements during model fine-tuning.

Swap should be used only in case of extreme necessity, as it can significantly slow down the training process. To ensure that the system uses swap space minimally, you should add the following line to the **/etc/sysctl.conf file**: **vm.swappiness=1**. This setting minimizes the swappiness, making the system less likely to swap processes out of physical memory and thus relying more on RAM, which is much faster than swap space.

## Convensions
- If a GPU is available, the Tuningtron library automatically leverages DeepSpeed to offload model weights to RAM. This optimization allows for efficient management of memory resources, enabling the fine-tuning of larger models even with limited GPU memory.
- The Tuningtron library supports only a specific dataset format, which must include the following columns: "instruct", "input", and "output". These columns are essential for the proper functioning of the library, as they structure the data in a way that the model can interpret and learn from effectively. If the dataset contains a column named "text", the library will use only this column and the data within it as-is.
- If the eval=True parameter is passed to the prepare_dataset method, the Tuningtron library will automatically use 10% of the data in the dataset as validation data, creating an evaluation dataset. This feature allows for easy splitting of the dataset, ensuring that a portion of the data is reserved for evaluating the model's performance during training, thereby facilitating better model assessment and tuning.
- The Tuningtron library fundamentally avoids using quantization during the fine-tuning process to prevent any potential loss of quality. This approach ensures that the experiments remain straightforward and maintain the highest possible model accuracy.
- For combining LoRA adapters, the Tuningtron library supports only the "cat" method. In this method, the LoRA matrices are concatenated, providing a straightforward and effective approach for merging adapters.

## Supported Models
The following LLM models are supported:
- Cohere Family
- Gemma Family
- Qwen Family

## SFT finetuning example
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)
os.environ["HF_TOKEN"] = "xxx"

tuner = Tuner("google/gemma-2-9b-it")
tuner.sft("equiron-ai/translator_sft", "adapter_gemma_sft", rank=64, batch_size=1, gradient_steps=1, learning_rate=1e-4)
```

## DPO finetuning example
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)
os.environ["HF_TOKEN"] = "xxx"

tuner = Tuner("./gemma_sft", enable_deepspeed=False)
tuner.dpo("equiron-ai/translator_dpo", "adapter_gemma_dpo", rank=64, batch_size=1, gradient_steps=1, learning_rate=1e-4)
```

## Combining/merging LoRA adapters
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import logging
from tuningtron import Tuner

logging.basicConfig(level=logging.INFO)

tuner = Tuner("google/gemma-2-9b-it")
tuner.merge("gemma_sft", "adapter_gemma_sft")
```

## Known issues
Model fine-tuning and combining adapters cannot be performed in the same bash script or Jupyter session. It is essential to separate the processes of fine-tuning and adapter merging. When using JupyterLab, you must restart the kernel after completing each of these processes to ensure proper execution and avoid conflicts.

## Convert to GGUF
```console
python3 llama.cpp/convert-hf-to-gguf.py /path/to/model --outfile model.gguf --outtype f16
llama.cpp/build/bin/llama-quantize model.gguf model_q5_k_m.gguf q5_k_m
```

## Run with Llama.CPP Server on GPU
```console
llama.cpp/build/bin/llama-server -m model_q5_k_m.gguf -ngl 99 -fa -c 4096 --host 0.0.0.0 --port 8000
```

## Install CUDA toolkit for Llama.cpp compilation
Please note that the toolkit version must match the driver version. The driver version can be found using the nvidia-smi command.
Аor example, to install toolkit for CUDA 12.4 you need to run the following commands:
```console
CUDA_TOOLKIT_VERSION=12-4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt -y install cuda-toolkit-${CUDA_TOOLKIT_VERSION}
echo -e '
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
' >> ~/.bashrc
```

