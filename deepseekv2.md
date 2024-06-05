# vllm运行deepseekv2推理

## 1. 切换到triplemu/deepseek-v2分支，源码编译安装vllm

```shell
git clone https://github.com/CC-LLM/vllm.git -b triplemu/deepseek-v2
cd vllm
pip install -r requirements-build.txt
pip install -r requirements-cuda.txt
pip install xformers # 预编译包不能用就需要源码编译了执行下面的命令
# pip install git+https://github.com/facebookresearch/xformers.git
pip install vllm-flash-attn --no-deps # 如果是nightly版本的pytorch需要加--no-deps避免覆盖安装pytorch

python setup.py install
python -c "import vllm; print(f'vllm version: {vllm.__version__}')" # 确认安装成功
```

## 2. vllm推理脚本配置

```python
import torch
from vllm import LLM, SamplingParams

# hack torch.load with mmap for accelerating
_load = torch.load
torch.load = lambda *args, **kwargs: _load(*args, **{**kwargs, 'mmap': kwargs.get('mmap', True)})

model = '/data/models/DeepSeek-V2'

llm = LLM(
    model=model,
    trust_remote_code=True,
    tensor_parallel_size=8,
    gpu_memory_utilization=0.95,
    enforce_eager=True,
    dtype='bfloat16',
    kv_cache_dtype='auto',
    max_model_len=2048,
)

sp = SamplingParams(temperature=0.3, max_tokens=256)

texts = [
    'hello, what is your name?',
    '大熊猫是什么？',
]

outputs = llm.generate(texts, sp, use_tqdm=False)
for i, output in enumerate(outputs):
    print(f'Q: {texts[i]} A: {output.outputs[0].text}')
```

目前已知deepseekv2推理存在问题，需要调整下面的参数：

- enforce_eager需要关闭，会有无法开启cudagraph的相关报错
- kv_cache_dtype无法应用fp8/fp8_e4m3/fp8_e5m2，会有xformers不支持的报错
- max_model_len需要设置小一点，模型默认的会导致推理时OOM


