(chap_gemm_basics)=
(chap_single_tile)=
(chap_k_loop)=
(chap_spatial_tiling)=
# 构建 Tiled GEMM

:::{admonition} 概览
:class: overview

- 从 TIRx tile 原语出发，从单个输出 tile 开始构建正确的 tiled GEMM。
- 第 1 步是单 tile 的 GEMM，第 2 步添加 K 循环累加，第 3 步在多个 CTA 上进行空间分块以覆盖完整矩阵。
- 正确性优先；性能是接下来两章的任务。
:::

GEMM 是贯穿本书的核心计算负载。它支撑了占据 GPU 大部分时间的线性层、注意力投影和卷积操作，因此一个正确的 GEMM 与一个快速的 GEMM 之间的区别，就是让芯片大部分闲置与让其饱和运行之间的区别。

这一差距无法一步跨越。一个饱和运行的内核会迫使您同时调试内存搬运、累加、分块和 Tensor Core 调度，而又没有任何可靠的对比基准。更安全的路径是从一个能产生正确结果的最小内核开始，然后逐步扩展。

本章将编写第一个正确的 tiled GEMM。前面的章节以抽象方式介绍了 TIRx 的 scope / layout / dispatch 模型；本章将其应用到真实的内核中。我们从单个 128 x 128 输出 tile 开始，逐步扩展成一个能处理完整尺寸矩阵的内核，依次添加 K 维度累加和在多个 CTA 上的空间分块。

这是贯穿单个 GEMM 优化路径的三个章节中的第一章。在本章中，我们构建一个正确的 tiled 内核并就此止步。下一章（{ref}`chap_gemm_async`）将线程拷贝替换为 TMA，并通过流水线技术使数据搬运与计算重叠，而 {ref}`chap_gemm_advanced` 则进一步引入 warp 专用化和 CTA 集群。每一章都建立在前一章的基础上，因此内核是逐步累加功能，而非重新开始。

将每一步看作是对一个包含三项约定的协定的编辑会更有帮助：哪个 **scope** 执行操作，操作数 tile 使用哪种 **layout**，以及哪个 **dispatch** 路径来执行它。大多数步骤只有一项主要变更，因此我们在每步开始时提供一个小卡片，说明该变更并指出任何使重用安全的必要同步细节。第 1 步为后续的优化路径建立了基线。

## GEMM

GEMM 是支撑线性层、注意力投影和许多卷积实现的密集矩阵乘法，这就是为什么一个快速的 GEMM 内核几乎在任何地方都能带来回报。本教程中的示例使用 $D = A B^{\top}$：

- $A$ 的形状为 $M \times K$。
- $B$ 的形状为 $N \times K$。
- $D$ 的形状为 $M \times N$。
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$。

转置并非我们选择执行的额外操作；它是由数据的存储方式自然决定的。示例将 $B$ 保持为 $N$ 行、每行长度为 $K$，这通常是线性层权重的存储布局，因此沿 $K$ 进行规约自然读取的是 $B^{\top}$，无需任何重排。

在本教程中，我们通过 TFLOPS 衡量内核的吞吐量，计算每次乘加的两个浮点操作相对于墙钟时间：

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{秒}} \times 10^{12}}$$

### GEMM 数据路径

本教程中的每一项优化归根结底都取决于数据所在的位置及其移动方式，因此在编写任何代码之前，先理清这一点是值得的。从根本上说，一个 Blackwell GEMM 内核仅围绕两个活动组织：在内存之间搬运 tile 以及对它们进行计算。下图追踪了一个 tile 从输入到输出所经过的每一级内存：

![*内存数据流*](../img/memory_dataflow.png)

上图展示了后续所有优化都会修改但从未替换的基线路径。
从左到右阅读：操作数 tile 首先从 GMEM 搬运到 SMEM；然后 `tcgen05.mma`
消费 SMEM 中的操作数并将累加器写入 TMEM；最后，后处理器将 TMEM 读回
寄存器，再将结果存储到 GMEM。请牢记这条链路，因为下面的每一步
都会改变这些跳跃中的*某一环*；但绝不会改变跳跃本身。

## 优化路径

上述基本数据路径足以得到正确的结果，但它让大部分硬件处于闲置状态。本教程的其余部分通过逐一添加 Blackwell 特性来缩小这一差距，每个特性都通过 TIRx tile 原语来表达。我们将依次遍历以下特性：

- **TMA 异步搬运** 通过 Blackwell 的硬件拷贝路径搬运 GMEM <-> SMEM tile，并使用屏障追踪完成状态。
- **软件流水线** 使用多个 SMEM 阶段，使得下一个 K tile 的数据搬运可以与当前 tile 的 Tensor Core 计算重叠。
- **持久调度** 维护一个固定的 CTA 池，每个 CTA 通过 tile 调度器处理多个输出 tile，而不是为每个 tile 启动一个 CTA。
- **Warp 专用化** 将生产者、MMA 消费者和写回角色分配到不同的 warpgroup 上。
- **CTA 集群** 允许两个 CTA 协作处理单个更大的 Blackwell MMA tile。
- **多消费者执行** 使用多个消费者 warpgroup 同时计算 tile 的不同部分，提高计算密度。

---

(chap_single_tile)=
## 第 1 步：顺序单 Tile GEMM

能完整走通全部硬件路径的最简单的 GEMM 是只计算一个输出 tile。这就是我们的起点。第 1 步计算一个 128 x 128 的输出 tile，K = 64，足够小以至于无需循环，数据路径的每一段都恰好出现一次。由于没有重复，我们可以在需要理解循环之前，先独立观察每一段跳跃。

> **本步建立的基础：基线**
> - Scope：一个由 128 个线程组成的 warpgroup 按顺序走完整条路径，一个阶段接一个阶段。
> - Layout：A 和 B tile 位于 SMEM 中，累加器位于 TMEM 中，结果通过寄存器分阶段写出。
> - Dispatch：同步的 `Tx.copy` 负责加载，`tcgen05` 负责执行 MMA。

### 单 Tile 数据流

基线约定确定后，接下来要确定的是一个 tile 遍历该路径的顺序。第一个内核恰好走一遍核心 GEMM 数据路径，即来自数据流图的同一条 GMEM -> SMEM -> TMEM -> 寄存器 -> GMEM 链路，没有任何循环包裹。它分配工作内存、加载操作数、计算乘积、写回结果，然后清理自身：

1. **分配**：SMEM（池分配器）、TMEM（`tcgen05.alloc`）、mbarrier
2. **加载**：全部 128 个线程协同将 A 和 B tile 从 GMEM 拷贝到 SMEM（同步 `Tx.copy`）
3. **计算**：单个选中的线程发起 `Tx.gemm_async` + `tcgen05.commit`；所有线程在 mbarrier 上等待
4. **写回**：Warpgroup 将 TMEM 读入寄存器；每个线程将 fp32 转换为 fp16 并写入 GMEM
5. **释放**：TMEM 释放

### 第一个内核的四个部分

完整的内核只有几十行，但分段理解更容易消化。我们将分四个部分（内存分配、同步加载、MMA 调度和写回）来阅读，之后再将它们组合成一个完整的内核。文中出现的 API 名称是第二部分（{ref}`chap_tirx_primer`、{ref}`chap_tirx_layout_api`）中介绍的 TIRx tile 原语词汇。

**内存分配。** 内核首先为操作数划分共享内存，以及一个用于 TMEM 地址的槽位和一个 mbarrier：

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM 地址 (4 字节)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 字节)
pool.move_base_to(1024)                           # 跳到偏移量 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

这里有两个细节值得注意。`pool.move_base_to(1024)` 将 Asmem 和 Bsmem 推至偏移量 1024 处，保留低地址用于上面的小型元数据，使得占用空间大的操作数 tile 落在干净的边界上。而 `layout=A_layout` 要求 `tma_shared_layout` 提供一个经过交错（swizzle）的 SMEM 布局，使 TMA 和 `tcgen05.mma` 都能直接读取，这正是第二部分所描述的布局即契约的约定。

**同步加载。** 缓冲区就位后，操作数仍需到达 SMEM。在这第一个版本中，我们让 CTA 自身的线程来完成拷贝：

```python
Tx.cta.copy(Asmem[:, :], A[:, :])
Tx.cta.copy(Bsmem[:, :], B[:, :])
T.cuda.cta_sync()
```

由于这里只有一个 tile（M=N=128, K=64），拷贝整个 A 和 B 就是全部的加载工作。`Tx.cta.copy(...)` 让 CTA 协同完成拷贝，每个线程负责自身的那部分数据。随后的 `T.cuda.cta_sync()` 有双重作用：它等待每个线程完成，并发布它们的共享内存写入，这样当 MMA 后续读取 `Asmem` 和 `Bsmem` 时，看到的是完整的 tile 而非半满的缓冲区。这种线程驱动的拷贝也是我们将要替换的第一件事；下一章（{ref}`chap_gemm_async`）将用 TMA 来替代它。

**MMA 调度。** 操作数现在位于 SMEM 中，我们可以发起 MMA 了，并由单个选中的线程来完成：

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

两层嵌套的守卫分两步缩小发起者的范围。外层 `if warp_id == 0` 只保留 warpgroup 中的 warp 0，内层 `if T.ptx.elect_sync():` 随后在该 warp 中选出一个活跃通道（lane）。两者共同确保恰好一个线程来执行 `Tx.gemm_async` 和 `tcgen05.commit`。

有必要澄清这个单线程究竟意味着什么和不意味着什么，因为直观理解可能具有误导性。单个发起线程并*不*意味着是单线程乘法。计算仍然是完整的 tile 级 MMA：硬件为 SMEM 操作数布局和 TMEM 累加器布局所描述的 tile 执行协同乘法。关键在于 `Tx.gemm_async` 是一个 *tile 操作*，而非一条硬件指令。K = 64 的 tile 比硬件 MMA K-atom（`MMA_K = 16`）更宽，因此这个单一的 tile op 会降低为沿 K 步进的一系列原始 `tcgen05.mma` 指令，而 warpgroup 协同驱动其中每一条指令。之所以只有一个线程发起 tile op，是因为每个底层的 `tcgen05.mma` 本身就是一个*单指令*协同操作：一次发射驱动该 tile MMA 的一个 K-atom。如果全部 128 个线程都发起这个序列，同样的工作将被简单重复发射 128 次。最后，`accum=False` 标志告诉 MMA 覆盖 TMEM 目标而非累加到其中，这正是我们这里需要的，因为没有先前的部分和需要扩展。

**写回。** 乘积现在位于 TMEM 中，但调用方希望它以 fp16 格式回到 GMEM 中。因此，后处理器必须通过寄存器将结果搬运下来并沿途完成类型转换：

```python
Dreg = T.alloc_local((BLK_N,), acc_type)        # 每线程 fp32 寄存器行
Dreg_f16 = T.alloc_local((BLK_N,), d_type)      # 同一行，转换为 fp16
Dreg_wg = Dreg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

MMA 在 TMEM 中留下了一个 128 x 128 的 fp32 累加器 tile。使用 fp32 是刻意的：GEMM 沿 K 累加了许多乘积，以更高精度保持累加和可以抑制本来会累积的舍入误差。但 `D` 是 fp16 的，因此数值不能直接输出。它们先落入寄存器，在那里缩窄为 fp16，然后才到达 GMEM。

两个寄存器缓冲区扮演不同的角色。`Dreg` 是每个线程的 `BLK_N` 元素缓冲区，而 `Dreg_wg` 是这些相同寄存器在选定布局下的 warpgroup 级*视角*：

```python
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

该布局将 tile 的第一维映射到 warpgroup 的线程上：线程 0 拥有行 0，线程 1 拥有行 1，依此类推直到行 127。第二维保留在每个线程自身的寄存器缓冲区内，因此单个线程持有其所在行的所有列。warpgroup 中有 128 个线程，tile 中有 128 行，128 x 128 的输出正好被划分为每线程一行。

在该视角下读取累加器正是 `Tx.wg.copy_async(Dreg_wg, tmem)` 所做的事情，它被降低为 Blackwell TMEM 加载路径 `tcgen05.ld`。由于该加载是异步的，`T.ptx.tcgen05.wait.ld()` 必须在任何线程访问 `Dreg` 之前完成；否则线程会读取加载尚未填充的寄存器。

等待返回后，每个线程私有的 `Dreg[:]` 持有其逻辑输出行的 fp32 值。该线程将这些值缩窄为 `Dreg_f16` 中的 fp16，然后计算出它负责的是全局的哪一行，

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
```

并写入 `D[m_thr, n_st:n_st + BLK_N]`。各线程行在四个 warp 之间清晰划分：warp 0 写入行 0-31，warp 1 写入行 32-63，warp 2 写入行 64-95，warp 3 写入行 96-127。

### 完整内核

现在我们将四个部分重新拼接成一个可运行的内核（M=N=128, K=64）。首先导入所需模块：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

内核被包装在后续步骤也使用的相同 `hgemm_vX(M, N, K)` 风格中。第 1 步使用 `M=N=128, K=64`，因此启动包含恰好一个输出 tile：

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K 记录了底层硬件 MMA tile 的尺寸；它们不会
    # 传递给 gemm_async（后者从操作数和累加器 tile 推导 MMA 形状），
    # 因此后续步骤省略它们。
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
        # 第 1 步是单 tile 内核：M = BLK_M 且 N = BLK_N，所以网格
        # 是 1x1。从 1x1 网格开始使得每个 CTA 的 tile 偏移量
        # (m_st, n_st) 简单为零；第 3 步及之后会将其推广到更大的 M / N。
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
    
        # --- 屏障 + TMEM 初始化（仅 warp 0） ---
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
    
        # --- 加载：所有线程将全局内存拷贝到共享内存（同步）。
        # 当 M=BLK_M 且 N=BLK_N 时，下面的切片覆盖了完整矩阵；
        # 保留切片形式是为了使与第 3 步（多 tile）的差异最小化。
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- 计算：单个选中的线程发起 MMA ---
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

后续的每一个 GEMM 步骤都以相同的方式编译、运行和自检，因此我们在此只完整展示一次这个脚手架代码，之后只显示内核本身。要运行后面的步骤，只需将下方的 `hgemm_vX` 和相应的矩阵大小替换即可。有一点值得注意：请在每次新鲜的 Python 会话中编译一个步骤，并在尝试另一个步骤之前重启，因为示例重用了内部名称，而编译器保有每次会话的状态。

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

# ex.mod(...) 直接接收 torch 张量，这是每章使用的相同调用形式。
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"与 torch 参考值的最大误差: {max_err:.6f}")
# 相对容差，与 warp 专用化和 Flash Attention 的单元格一致：
# 输出量级随 K 增长，因此固定的绝对边界在较大的 K 时会失败。
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")

# 可选：对更大内核的计时。
ITERS = 10
for _ in range(3):
    ex.mod(A_tensor, B_tensor, D_tensor)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    ex.mod(A_tensor, B_tensor, D_tensor)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"性能: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

第 1 步到第 3 步故意使用较小的大小（此处为 128×128，第 3 步为 256³）以保持这些初次演练简单易懂。{ref}`chap_gemm_advanced` 末尾的跨步骤*端到端结果*表格则采用相反的方式：它在单一的 M=N=K=4096 大小上测量每一步（包括这里的第 1 步算法），使得加速比可以直接比较。

### 单 Tile 内核的局限性

这个内核是正确的，这正是第 1 步的全部目标，但它仅在一个非常狭窄的范围内正确。我们有意内置了四个限制，后续的优化路径将逐一解除：

- 它只处理单个 K tile，因此无法对较大的 K 进行规约。
- 它只处理单个输出 tile，因此 M 和 N 被固定在 128。
- 它使用同步的 GMEM -> SMEM 拷贝，而非 TMA。
- 它没有重叠数据搬运与计算，因此两者从未同时运行。

---

(chap_k_loop)=
## 第 2 步：K 循环累加

首先要解除的是最小的那个限制。第 1 步只处理单个 64 宽的 K tile，而真实的矩阵在 K 维度上要远大于此。在第 2 步中，我们保持单个输出 tile，但让 K 跨越多个 64 宽的块。

思路很直接：对每个块重复执行加载 -> MMA -> 等待序列，并让每次 MMA 累加到同一个 TMEM 槽位中。真正的难点在于同步。跨迭代重用一个 mbarrier 引入了本章第一个真正的正确性隐患。如果代码追踪了错误的 phase，一次等待可能会在其 MMA *尚未* 实际完成时就返回，从而静默地破坏结果。下面的机制细节将精确展示这如何出错，以及如何避免。

> **本步变更的内容：布局重用**
> - Scope：不变，仍为单个 warpgroup。
> - Layout/重用：相同的 SMEM tile 对和 TMEM 累加器槽位在 K 循环中重复使用。不分配新的存储空间；操作数 tile 流经一对固定的缓冲区，累加器状态则驻留在同一个 TMEM 槽位中。
> - 同步：被重用的 MMA 屏障必须在每个 K 块上推进到正确的 phase，否则后续的等待可能观察到较早一次完成的信号。
> - Dispatch：不变。

### K 循环机制

第 1 步仅在一个 64 宽的 K tile 上进行规约；这里我们保持其单个输出 tile，但让 K 按照矩阵所需任意延伸。为了覆盖大于 64 的 K，我们以 `BLK_K=64` 的块为单位遍历 K。每次迭代加载下一个 A 和 B 的 K 切片到 SMEM 中，并发起 `Tx.gemm_async`。`accum` 标志将这些块拼接成一个完整的点积：在第一个块上，`accum=False` 初始化 TMEM 累加器，而在后续每个块上，`accum=True` 将该块的乘积累加到已存在于 TMEM 中的运行和上。

同步是需要小心的地方。我们为每次 MMA 完成重用单个 mbarrier，而安全重用的关键在于追踪我们在等待哪个屏障 phase。一个 mbarrier 携带 1 比特的 phase，值为 0 或 1，并在预期到达次数满足后翻转为另一个值。微妙之处在于等待条件本身：`try_wait(bar, phase)` 会阻塞，直到屏障的内部 phase *不同于* `phase` 参数。因此，我们传入的参数必须指定我们期望*离开*的 phase，而非我们正在等待*到达*的 phase：

| K 迭代 | 等待前的本地 `phase_mma` | `try_wait` 等待的内容 | 等待后的本地更新 |
|---|---:|---:|---:|
| 0 | 0 | 屏障翻转为 1 | `phase_mma = 1` |
| 1 | 1 | 屏障翻转为 0 | `phase_mma = 0` |
| 2 | 0 | 屏障翻转为 1 | `phase_mma = 1` |

单行代码 `phase_mma ^= 1` 正是保持上述表格正确的关键。如果去掉它，第二次迭代仍然调用 `try_wait(bar, 0)`，但屏障在第一次 MMA 后已经翻转为 phase 1，因此等待看到不匹配而立即返回，此时第二次 MMA 尚未完成。内核随后读取一个半计算的累加器并报告错误的结果，且没有任何报错。这是一个可以完美编译和运行的 bug，这正是 phase 翻转值得如此关注的原因。

### 完整内核

下面的完整内核就是第 1 步加上 K 循环和 phase 翻转。导入模块与之前相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

它被包装在 `hgemm_v2(M, N, K)` 中。网格仍然是 `[1, 1]`，因为我们仍在计算单个输出 tile；增长的部分只有 K 的范围：

```python
def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])  # 仍然只有一个输出 tile (M=N=128)
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # === K 循环：以 BLK_K 为步长遍历 K ===
        for i in T.serial(K_TILES):   # serial 设备循环（保持完整 K 的 A/B 参数形状正确）
            # 加载第 i 个 K 块
            Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            # MMA：第一个 tile 使用 accum=False，其余使用 accum=True
            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # 等待 MMA，然后翻转 phase
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # === 写回（与第 1 步相同） ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

---

(chap_spatial_tiling)=
## 第 3 步：空间分块（多 CTA）

K 循环解决了规约维度的问题，但 M 和 N 仍然被固定在一个 128 x 128 的 tile 上。真实的输出远大于一个 tile，因此基础内核的最后一块是同时用多个 tile 覆盖 M 和 N。第 3 步启动一个二维 CTA 网格，每个输出 tile 对应一个 CTA，让 GPU 并行计算所有 tile。示例使用 M=N=K=256，产生一个 2x2 的 tile 网格，恰好使索引变得非平凡但又不至于让人迷失。

> **本步变更的内容：Scope**
> - Scope：二维 CTA 网格，每个 CTA 拥有一个 128 x 128 的输出 tile。
> - Layout：不变；在 CTA 内部，与第 2 步相同的 SMEM/TMEM/寄存器路径。
> - Dispatch：不变。

### 网格映射

网格形状直接由分块决定：每个 128 x 128 输出 tile 对应一个 CTA，总共需要 `[M // BLK_M, N // BLK_N]` 个 CTA。与第 2 步相比，唯一真正的新工作是让每个 CTA 知道矩阵的哪一部分是*它的*要计算的部分。

CTA `(bx, by)` 拥有以下输出区域：

```text
D[bx * BLK_M : (bx + 1) * BLK_M,
  by * BLK_N : (by + 1) * BLK_N]
```

为了产生这个输出，该 CTA 的 K 循环重复加载其自身的 A 行带和 B 列带中对应的 K 切片：

```text
A[bx * BLK_M : (bx + 1) * BLK_M, k : k + BLK_K]
B[by * BLK_N : (by + 1) * BLK_N, k : k + BLK_K]
```

索引直接遵循 `D = A @ B.T` 约定：`bx` 选择 A 和 D 的行，而 `by` 选择 B 的行，经过转置后它们成为 D 的列。

每个 CTA 一个 tile 是最简单的可行映射，但也是浪费的。同一行中的每个 CTA 都会从 GMEM 重新加载相同的 A tile，同一列中的每个 CTA 也会重新加载相同的 B tile，因此没有任何数据被相邻 CTA 已经拉取的数据重用。我们暂时保留这种浪费；持久调度（{ref}`chap_gemm_async` 中的第 6 步）会回来解决这个问题，并将这些共享操作数保留在 L2 中。

**尝试与您的智能体互动**：使用 `M=N=K=256`、`BLK_M=BLK_N=128` 和 `BLK_K=64`，让智能体追踪 CTA `(1, 0)` 和 CTA `(0, 1)`。对每个 CTA，列出 `m_st`、`n_st`、每次 K 迭代加载的 A 和 B 切片，以及写入的 D 区域。由于内核计算的是 `D = A @ B.T`，哪些 B 行变成了 D 的列？

### 完整内核

内核再次基于第 2 步，这次只做了两处改动：网格形状和每个 CTA 的偏移量。内部的 K 循环和写回保持不变。导入模块相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

网格变为 `[M // BLK_M, N // BLK_N]` 而非 `[1, 1]`，加载和存储现在由 CTA 自身的 `m_st` 和 `n_st` 进行偏移：

```python
def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 2D 网格：每个 128x128 输出 tile 对应一个 CTA
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0

        # 每 CTA 的 tile 偏移量
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # 带有偏移 A 和 B 切片的 K 循环
        for i in T.serial(K_TILES):   # serial 设备循环（保持完整 K 的 A/B 参数形状正确）
            Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # 写回到正确的输出 tile
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## 练习题

1. 在第 1-3 步中，`Tx.copy` 在 MMA 之前将 A 和 B tile 搬运到 SMEM。为什么内核需要在 `Tx.gemm_async` 读取这些 SMEM tile 之前调用 `T.cuda.cta_sync()`？
2. 在第 2 步中，如果从 K 循环中移除 `phase_mma ^= 1`，会发生什么？内核会等待每次 MMA 完成，还是后续的等待可能过早通过？
3. 对于 M=N=4096，BLK_M=BLK_N=128，第 3 步会启动多少个 CTA？哪些操作数 tile 在逻辑上被相邻的 CTA 重用，第 3 步是否利用了这种重用？
