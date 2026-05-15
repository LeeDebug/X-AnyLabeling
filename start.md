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





---

## XXX

```bash
# xxx
xxx
```
