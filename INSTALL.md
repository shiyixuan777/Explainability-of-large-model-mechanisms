# Installation Guide

## 1. Install Python

当前机器的 PowerShell 中没有可用的 `python` 或 `py` 命令。建议安装 Python 3.11 或 3.12，并在安装时勾选 **Add Python to PATH**。

安装后重新打开 PowerShell，检查：

```powershell
python --version
pip --version
```

## 2. Create a Virtual Environment

```powershell
cd D:\可解释性
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果 PowerShell 不允许激活虚拟环境，运行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Install Project Dependencies

CPU 版本：

```powershell
pip install -r requirements.txt
```

如果你有 NVIDIA GPU，建议先去 PyTorch 官网确认与你 CUDA 版本匹配的安装命令，再安装 `transformer-lens` 等依赖。

## 4. Check Environment

```powershell
python -m scripts.check_env
```

## 5. First Run

```powershell
python -m scripts.build_dataset
python -m scripts.build_balanced_capital_dataset
python -m scripts.run_probe --model gpt2-small --data data/facts.csv --language en --domain capital --prompt-template "Statement: {statement}`nAnswer true or false:" --out figures/probe_capital_answer.csv
```

第一次运行会下载模型权重，需要稳定网络。`gpt2-small` 体积较小，适合先跑通全流程。
