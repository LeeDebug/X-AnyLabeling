# 配置命令

## 卸载命令

```bash
# 删除虚拟环境（退出虚拟环境后执行）
deactivate
rm -r xxx
```

## 创建并激活虚拟环境

```bash
# 创建虚拟环境（指定 Python 3.11，兼容性最好）
uv venv --python 3.11

# 激活虚拟环境
.venv\Scripts\activate
# or
.\.venv\Scripts\activate
```

## 安装 GPU 版本 X-AnyLabeling

```bash
# 查看本地
nvcc --version
nvidia-smi

# 官方推荐命令（自动处理 PyTorch 和 ONNX Runtime GPU 依赖）
uv pip install x-anylabeling-cvhub[gpu]
```

## 验证 GPU 环境是否正常

```bash
# 检查 X-AnyLabeling 安装
xanylabeling checks
```

## 启动 X-AnyLabeling

```bash
# 直接启动
xanylabeling

# 或者指定工作目录启动
xanylabeling --output ./labels
```


## 从源码安装（安装开发者模式依赖）

```bash
# 1. 安装与您驱动完美匹配的CUDA 12.6版本PyTorch
uv pip install torch torchvision torchaudio --torch-backend=cu126

# 输出如下
# Resolved 14 packages in 787ms
# Prepared 3 packages in 6m 21s
# ░░░░░░░░░░░░░░░░░░░░ [0/9] Installing wheels...                                                                                                                  warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.                                                             
#          If the cache and target directories are on different filesystems, hardlinking may not be supported.
#          If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
# Installed 9 packages in 19.12s
#  + jinja2==3.1.6
#  + markupsafe==3.0.3
#  + mpmath==1.3.0
#  + networkx==3.6.1
#  + setuptools==81.0.0
#  + sympy==1.14.0
#  + torch==2.12.0+cu126
#  + torchaudio==2.11.0+cu126
#  + torchvision==0.27.0+cu126

# 让 uv 自动检测最优版本（最省心）
# uv pip install torch torchvision torchaudio --torch-backend=auto

# 2. 安装训练功能核心依赖 ultralytics
# 必须在 PyTorch 之后安装，这样它会自动使用已安装的 GPU 版本 PyTorch
uv pip install ultralytics==8.3.60
# 输出如下
# Installed 7 packages in 2.03s
#  + opencv-python==4.13.0.92
#  + pandas==3.0.3
#  + py-cpuinfo==9.0.0
#  + seaborn==0.13.2
#  + tzdata==2026.2
#  + ultralytics==8.3.60
#  + ultralytics-thop==2.0.19

# 3. 安装X-AnyLabeling完整训练依赖
uv pip install -e .[train,gpu]

# CUDA 11.x
# uv pip install -e .[gpu-cu11]

# 4. 验证所有依赖安装
python -c "
import torch
import ultralytics
import onnxruntime as ort

print(f'✅ PyTorch 版本: {torch.__version__}')
print(f'✅ CUDA 可用: {torch.cuda.is_available()}')
print(f'✅ Ultralytics 版本: {ultralytics.__version__}')
print(f'✅ ONNX Runtime GPU 版本: {ort.__version__}')
print('\n🎉 所有依赖安装成功！可以开始使用标注和训练功能了')
"
# 输出如下
# ✅ PyTorch 版本: 2.12.0+cu126
# ✅ CUDA 可用: True
# ✅ Ultralytics 版本: 8.3.60
# ✅ ONNX Runtime GPU 版本: 1.26.0
# 🎉 所有依赖安装成功！可以开始使用标注和训练功能了

# 5. 启动程序
python anylabeling/app.py
```




---

## XXX

```bash
# xxx
xxx
```
