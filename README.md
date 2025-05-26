# 安装环境
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir mllm_learning
cd mllm_learning
uv venv .venv --python=python3.13
source .venv/bin/activate
uv init
uv sync
uv add hydra-core
uv add torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index https://download.pytorch.org/whl/cu118
```

修改.env里面的PYTHONPATH为当前主目录
