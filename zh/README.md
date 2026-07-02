# 面向机器学习系统的现代 GPU 编程

本书以递进的方式教授现代 GPU kernel 编程：**理解
GPU 硬件 → 学会编程它 → 编写当前最优的 kernel。**它将
Blackwell 级 GPU——其内存层次和张量内存、其张量核心和
异步数据搬运引擎、warpgroup 和 cluster——作为真正的主题。编程
载体是 **TIRx**（Tensor IR next），一个用于在 IR 级别编写 GPU kernel 的 Python DSL。

📖 **在线阅读：<https://mlc.ai/modern-gpu-programming-for-mlsys/>**

**中文版：<https://mlc.ai/modern-gpu-programming-for-mlsys/zh/>**

🤝 **贡献：** 欢迎通过
[GitHub 仓库](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys)提交更正、示例和改进。

## 内容概要

- **第一部分 — 理解 GPU。** 执行和内存模型，性能模型
  （roofline、重叠），深入探讨数据布局，内存和计算引擎（TMA、
  张量内存、张量核心），异步协调，以及高级调度（CLC）。
- **第二部分 — 用 TIRx 编程 GPU。** 通过一个可运行的单 MMA GEMM 介绍
  TIRx——范围、布局和分发，以及编译的工作原理——加上张量
  布局模型（`TileLayout`、命名轴、交错）。
- **第三部分 — GEMM：从 Tiled 到当前最优。** 一个 tiled GEMM 通过 TMA 流水线、
  持久化调度、warp specialization 和 2-CTA cluster 逐步构建。
- **第四部分 — Flash Attention 4。** 一个基于第三部分技术构建的完整 attention kernel：
  两个 MMA 中间夹着 softmax，在线 softmax 重缩放，因果 masking 和 GQA。
- **参考。** TIRx 语言参考和编译器内部。

## 本地构建本书

本书是一个 [Sphinx](https://www.sphinx-doc.org/) 站点（Markdown/MyST + reStructuredText）：

```bash
pip install -r requirements-docs.txt
sphinx-build -b html . _build/html
```

### 预览

```bash
python -m http.server -d _build/html 8000
```

打开 <http://localhost:8000>。在远程机器上，服务器在那里运行，所以转发
端口 — `ssh -L 8000:localhost:8000 user@your-server` — 然后在本地打开 URL。（VS Code
Remote SSH 会自动转发它。）

## 运行 Kernel（需要 Blackwell GPU）

本书中的 kernel 面向 Blackwell（`sm_100a`），所以运行它们需要 Blackwell GPU
（例如 B200）、TIRx 编译器和一个 CUDA 构建的 PyTorch。

**1. 安装 TIRx 编译器。** 它作为 Apache TVM wheel 的 `tvm.tirx` 模块发布：

```bash
pip install apache-tvm
```

验证：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__version__)"
```

**2. 安装 PyTorch**，使用与你的 GPU 匹配的 CUDA 构建（用于示例输入和
参考检查）— 参见 <https://pytorch.org>。

**3.（可选）参考 kernel。** 完整的 GEMM 和 Flash Attention 4 kernel 位于
配套的 `tirx-kernels` 包中（从 checkout `pip install -e .`）；使用例如
`python -m tirx_kernels.test --kernel fp16_bf16_gemm` 运行它们。

TIRx 通过 Python 源代码检查来解析 kernel 源，因此示例应该放在文件
或 notebook 单元中，而不是 `python -c` 中。

## 部署

每次向 `main` 推送都会由 GitHub Actions
（`.github/workflows/build_deploy.yaml`）自动构建并发布到
<https://mlc.ai/modern-gpu-programming-for-mlsys/>。
