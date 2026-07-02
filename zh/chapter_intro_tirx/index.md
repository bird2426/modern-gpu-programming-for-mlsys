(chap_tirx_primer)=
# TIRx 入门

:::{admonition} 概览
:class: overview

- TIRx 是一个用于在 IR 层面编写 GPU kernel 的 Python DSL：你直接命名硬件，但是通过结构化的 IR。
- 每个 tile 操作由三个设计元素控制：*范围*（哪些线程）、*布局*（tile 在哪里）和*分发*（哪个硬件路径）。
- 一个可运行的单 MMA GEMM 展示了全部三者；本书的其余部分是这些设计元素的规模化应用。
:::

:::{admonition} 运行示例
:class: note

这些示例需要 Blackwell GPU（`sm_100a`，例如 B200）。TIRx 编译器作为 Apache TVM wheel 的
`tvm.tirx` 模块发布；与 PyTorch 的 CUDA 构建一起安装：

```bash
pip install apache-tvm
```

通过 `python -c "import tvm, tvm.tirx; print(tvm.__version__)"` 确认导入成功。相同的设置
可以运行书中所有可运行的示例。
:::

第一部分解释了硬件是什么。要让它计算任何东西，我们需要一种编程它的方式。

我们可以写原始的 CUDA 或 PTX，许多快速的 kernel 正是这样写的。问题在于，实际决定 kernel 行为的那些决策在那里很难看清：哪些线程运行一个操作，每个数据 tile 在哪里，以及哪个硬件路径执行它。这些选择被埋藏在内在函数参数、地址算术和约定中。

TIRx（Tensor IR neXt）是一个 Python DSL，将这三个决策提升到明面上：**范围**（哪些线程运行一个操作）、**布局**（操作数 tile 在哪里）和**分发**（哪个硬件路径执行它）。它仍然直接命名硬件概念，包括线程、共享和张量内存、barrier 和 `tcgen05` MMA。区别在于，这些选择现在是编译器可以降低、检查和调度的结构化 IR。

我们不会抽象地介绍这些概念，而是从一个完整的 kernel 开始：一个最小的单 MMA GEMM。我们先让它运行起来，然后再逐行回顾，看看范围、布局和分发各自如何塑造它，以及 kernel 如何编译。kernel 所依赖的张量布局模型在 {ref}`chap_tirx_layout_api` 中独立展开，完整的语言特性集在 {ref}`chap_language_reference` 中；这里我们将焦点保持在这一个 kernel 和三个设计元素上。

## 第一个 Kernel：单 MMA GEMM

我们承诺的 kernel 是一个最小的 GEMM，精简到仍然能使用 Tensor Core 的最小版本。它计算 `D = A B^T` 的单个 128 x 128 输出 tile，其中 K = 64。整个计算表示为一个 `Tx.gemm_async` tile 操作，端到端。（那一个 tile 操作并不映射到单条硬件指令：因为硬件 MMA K-atom 是 16，K=64 的 tile 会降低为一系列沿 K 步进的 `tcgen05.mma` 指令。DSL 的意义正是我们写 tile，而不是指令序列。）围绕该操作，kernel 做通常的杂务：分配共享内存（SMEM）和张量内存（TMEM），将 A 和 B 从全局内存拷贝到共享内存，将 tile MMA 发射到 TMEM 累加器中，通过寄存器读回该累加器，并存储结果。虽然小，但这个 kernel 是我们在 {ref}`chap_gemm_basics` 中攀登的 GEMM 阶梯的第 1 步，在那里它会带着完整的逐步讲解返回。

每个 TIRx kernel 都从相同的少量导入开始，所以值得在前面一次性看到它们：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

我们将 kernel 包装在一个小的构建器 `hgemm_v1(M, N, K)` 中，它接受问题形状并返回一个 `PrimFunc`。对于我们选择的形状 `M=N=128, K=64`，启动恰好包含一个输出 tile，这使得第一个版本简单到可以一次读完：

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K 记录了底层硬件 MMA tile 的尺寸；它们不
    # 传递给 gemm_async（后者从操作数和累加器 tile 推导出 MMA 形状），
    # 所以后续步骤省略它们。
    MMA_M, MMA_N, MMA_K = 128, 128, 16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 第 1 步是一个单 tile kernel：M = BLK_M 且 N = BLK_N，所以 grid
        # 是 1x1。以 1x1 grid 开始使得每个 CTA 的 tile 偏移量
        # (m_st, n_st) 天然为零；第 3 步及以后将此推广到更大的 M/N。
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # 单 warpgroup，所以 wg_id 始终为 0（下面未使用）
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM 分配 ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
    
        # --- Barrier + TMEM 初始化（仅 warp 0） ---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
    
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
    
        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )
    
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0
    
        # --- 加载：所有线程拷贝 global -> shared（同步）。
        # 当 M=BLK_M 且 N=BLK_N 时，下面的切片覆盖完整矩阵；
        # 保留切片形式是为了让到第 3 步（多 tile）的 diff 最小。
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- 计算：单个选中线程发出 MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    
        # --- 写回：TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
    
        # --- 释放 TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

在阅读 kernel 之前，让我们先确认它能工作。我们编译它并对照 torch 参考检查其输出。我们不需要显式指定确切的架构：架构（例如 `sm_100a`）从设备自动检测，所以目标 `"cuda"` 就足够了，`tir_pipeline="tirx"` 选择 TIRx 降低流水线。编译后，`ex.mod(...)` 直接接受 torch tensor，中间不需要手动转换。

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device('cuda')  # gpu(0)

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

# ex.mod(...) 直接接受 torch tensor，所有章节使用相同的调用形式。
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")
```

## 范围、布局、分发

现在 kernel 可以运行了，我们可以回顾它并问它的各行实际决定了什么。从这个角度看，整个 kernel 是沿着三个设计元素的一组选择。其中的每个操作都回答相同的三个问题：*谁*运行它，它的数据在*哪里*，以及它*如何*执行，这三个答案正是范围、布局和分发。本节的其余部分逐一讨论设计元素；下面的交互式演示让你看到每个设计元素控制哪些行。

```{raw} html
<iframe src="../demo/tirx_dispatch.html" title="TIRx：范围、布局、分发" loading="lazy"
        style="width:100%; min-width:960px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互式：点击 Scope / Layout / Dispatch 来高亮每个设计元素控制的 kernel 行。*

当你使用演示时，留意三个问题：

- **范围：谁运行操作？** `Tx.cta.copy(...)` 是 CTA 范围的，所以所有 128 个线程协助 GMEM -> SMEM 拷贝。`Tx.gemm_async(...)` 由一个选中线程发出一次，因为每条降低后的 `tcgen05.mma` 指令已经是协作 MMA 启动。`Tx.wg.copy_async(...)` 是 warpgroup 范围的，所以 warpgroup 的 128 个线程逐行拆分 TMEM 回读。
- **布局：每个 tile 在哪里？** A 和 B 使用 `tcgen05.mma` 期望的交错 SMEM 布局。累加器驻留在 TMEM 中，使用 `TLane`/`TCol` 布局。寄存器回读视图将行映射到 `tid_in_wg` 上，因此每个 warpgroup 线程拥有一个行片段。
- **分发：哪个硬件路径执行它？** `Tx.gemm_async(..., dispatch="tcgen05", ...)` 选择 Blackwell Tensor Core 路径。拷贝操作也有分发选择：第一个 kernel 使用普通线程拷贝，后续 GEMM 步骤在不改变周围范围或布局的情况下将这些拷贝替换为 TMA。

**与你的 agent 一起尝试**：从第一个 kernel 中选三行：一个拷贝、一个 MMA 和一个 TMEM 回读。让它按范围、布局和分发标注每行，然后检查答案是否与代码中的守卫、缓冲区布局和 `dispatch=` 参数匹配。

## 编译如何工作

我们已经在上面编译了 kernel 来测试它；现在我们更仔细地看看那一步做了什么。配方很短：将 `PrimFunc` 包装在 `IRModule` 中并交给 `tvm.compile(mod, target=..., tir_pipeline="tirx")`。这运行 TIRx 降低流水线并返回一个可以直接调用的 `Executable`。

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

值得知道，至少在大致轮廓上，`tir_pipeline="tirx"` 启动了哪些流程。流水线的核心 pass `LowerTIRx` 对照其范围/布局/分发契约解析每个 tile 原语：这就是我们刚刚讨论的三个设计元素实际被兑现为指令的地方。之后，通常的主机/设备拆分和一个最终化步骤产生可启动的模块。如果你愿意，也可以在 `with target:` 块内编译，这让 kernel 获取周围的目标上下文。

这个流程的一个好处是没有东西对你隐藏：结果可以在两个层面检查。你可以用 `.show()` 或 `.script()` 阅读 IR 本身，也可以直接从编译后的模块阅读编译器最终发出的 CUDA C。

```python
kernel.show()                          # 美化打印 TIRx（TVMScript）
print(kernel.script())                 # ... 相同内容，作为字符串

# 生成的 CUDA C 源代码，来自编译后的 Executable：
print(ex.mod.imports[0].inspect_source())
```

这只是一个概要。完整的降低过程，涵盖所有 pass、tile 原语分发如何解析以及主机/设备拆分如何完成，参见 {ref}`chap_arch`。

## 下一步去哪里

一个 kernel 足以认识范围、布局和分发，并看到它们被编译和运行。这三个设计元素中的每一个，以及 kernel 本身，都开启了一个进一步展开的章节：

- {ref}`chap_tirx_layout_api`：张量布局模型（`TileLayout`、命名轴、交错），上面的操作数和累加器放置正是建立在它之上。如果布局设计元素是三者中最神秘的，从这里开始。
- {ref}`chap_language_reference`：完整的语言特性集，涵盖解析器工具、数据类型、缓冲区和内存、控制流以及线程同步，当你想要完整的词汇表而不是导览时。
- {ref}`chap_gemm_basics`：此 kernel 作为 GEMM 优化路径的第 1 步，通过 K 循环累加、空间 tiling、TMA 和 warp specialization 逐步建立。如果你想看到相同的三个设计元素扩展到真正的 kernel，这是自然的下一站。
