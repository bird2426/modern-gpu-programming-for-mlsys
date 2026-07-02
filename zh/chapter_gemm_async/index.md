(chap_gemm_async)=
(chap_tma_async)=
(chap_software_pipeline)=
(chap_persistent_kernel)=
# 用 TMA 为 GEMM 建立 Pipeline

> 翻译状态：已完成。对应英文章节：`chapter_gemm_async/index.md`。

:::{admonition} Overview
:class: overview

- 基础的 GEMM 在轮流执行时浪费了时间（拷贝一个 tile，计算，再拷贝下一个），而这两者本可以同时进行。
- 第 4 步改用 TMA 异步加载，第 5 步对 SMEM 进行双缓冲和预取（PIPE_DEPTH=2）；完整的加载/计算重叠在第 7 步的 warp 特化中实现，第 6 步通过 tile 调度器使 kernel 变为持久化。
- 目标是在 Tensor Core 处理当前 tile 的同时加载下一个 tile。
:::

Tensor Core 是芯片上最昂贵的单元，而前一章中正确的分块 GEMM 让它们在大部分时钟周期内处于空闲状态。kernel 轮流执行：线程将 tile 拷贝到共享内存，Tensor Core 处理它，线程再拷贝下一个 tile，而 Tensor Core 则等待。每个阶段都在等待前一个阶段，尽管加载下一个 tile 和计算当前 tile 使用的是完全独立的硬件，且可以同时运行。消除这一间隙并不需要新的数据通路；tile、布局和数学计算都是正确的。需要改变的是工作*何时*执行以及*由谁*调度。本章保持 tile 数据通路完全不变，直接解决空闲问题。

我们通过三个递增的步骤来实现目标，在开始之前先了解最终目标会有所帮助。在第 4 步中，我们将 GMEM <-> SMEM 之间的大量传输交给 TMA，这样专用的拷贝硬件（而不是线程）来移动 tile。在第 5 步中，我们添加一个两阶段的软件流水线，为下一个 K tile 提供落地空间，同时当前 tile 仍在进行乘法运算。在第 6 步中，我们将启动方式重塑为由 tile 调度器驱动的持久化 kernel，从而分摊每个 tile 的设置开销，并让我们能够选择保持操作数热的 tile 顺序。整个过程，SMEM、TMEM 和寄存器的布局与前一章完全一致。唯一真正的新概念是硬件单元之间的异步交接：让一个引擎领先于另一个引擎运行，而不是让它们步调一致地前进。

(chap_tma_async)=
## 第 4 步：TMA 异步加载

我们的第一步是将拷贝本身移出关键路径。想想 CTA 在第 1-3 步中做了什么：它的每个线程都在计算地址并发出加载指令，而这一切只是为了将 tile 搬运到 SMEM 中。这是花在管道工作上而非数学计算上的指令带宽。第 4 步用 TMA 替换了同步的 `Tx.copy`，其中单个线程发出一个命令，TMA 引擎自行完成整个 tile 的传输。从此处开始，示例运行在完整的 M=N=K=4096 尺寸上，而非第 1-3 步的小尺寸，它们的端到端时间出现在 {ref}`chap_gemm_advanced` 末尾的*端到端结果*表中。

> **此步骤更改的内容：调度**
> - 范围：不变，一个 warpgroup。
> - 布局：不变，相同的 SMEM/TMEM/寄存器 tile。
> - 调度：GMEM → SMEM 加载从同步的 `Tx.copy` 迁移到 TMA 引擎。

### TMA 下发模式

第 4 步的唯一变化是将同步的 tile 拷贝替换为 TMA 加载，因此值得仔细看看这个加载是如何下发的。对源码的修改只有几行，但这些行背后的执行模型在本质上是不同的。同步的 `Tx.copy` 是 CTA 线程用自己的指令完成的工作；而 TMA 拷贝是一个线程发出的命令，之后 TMA 硬件完成所有的搬运工作。将两者放在一起对比是值得的。

**之前（第 3 步）**：所有 128 个线程参与拷贝，然后 `cta_sync` 使共享内存写入可见：
```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])   # all 128 threads
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()
```

**之后（第 4 步）**：一个线程发出 TMA 加载，mbarrier 跟踪硬件传输何时完成：
```python
tid = warp_id * 32 + lane_id                 # 0..127 within the warpgroup
if tid == 0:  # exactly one thread starts TMA
    Tx.copy_async(Asmem, A[...], dispatch="tma")
    Tx.copy_async(Bsmem, B[...], dispatch="tma")
    T.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)  # bytes expected from TMA
T.ptx.mbarrier.try_wait(tma_bar, phase)                  # wait before MMA reads SMEM
```

注意加载是以 `tid == 0` 为条件的，而不是 `elect_sync()`，这个区别比看起来更重要。`elect.sync` 每个 warp 选举一个活跃通道，而一个 warpgroup 有四个 warp，因此 `elect_sync()` 实际上会让四个线程进入加载协议。问题在于该协议会向 mbarrier 宣告预期的字节数，并且必须恰好宣告一次；四次宣告会破坏计数，导致等待永远无法正确释放。通过 warpgroup 全局 ID 精确选择一个线程是避免这一问题的干净方式。

诚实地说明加速的来源是很重要的。第 4 步在每次 TMA 加载后仍然会等待，因此我们还没有将加载与计算重叠起来；那是第 5 步的工作。这里的收益纯粹来自数据移动路径的改变：

- `Tx.copy` 使用 CTA 线程计算地址并发出加载/存储指令。
- TMA 使用一个发出的命令启动硬件 tile 传输。地址生成、合并和 swizzle 由 TMA 描述符描述并由 TMA 引擎执行。

因此，尽管第 4 步仍然在每次加载时阻塞，但它最终还是会更快。TMA 吸收了大量传输，从而释放了 CTA 线程，使它们不必花费指令带宽来搬运 tile，仅这一项节省就足以带来明显的改善。

### TMA 加载与存储同步

我们已经看到了 TMA 拷贝是如何下发的；另一半故事是知道它何时完成。切换到 TMA 同时改变了两件事：谁启动拷贝，以及代码如何知道拷贝何时完成。第一点从代码中显而易见；第二点容易被忽视，而弄错了会给你带来一个静默的正确性错误而不是崩溃。使用 `Tx.cta.copy` 时，CTA 线程一起完成拷贝，随后的 `cta_sync()` 就足以知道拷贝已完成。使用 TMA 时，一个选定的线程发出 `Tx.copy_async(..., dispatch="tma")`，引擎按照自己的调度执行传输，并通过 mbarrier 发出完成信号。

这正是 `cta_sync()` 不再足够的原因。`cta_sync()` 只等待 CTA 自身的线程，并且只对它们的共享内存写入进行排序；它对正在进行的 TMA 传输一无所知，因此它会在 tile 仍在到达时就愉快地返回。解决办法是使完成变得显式：对于 TMA 加载，选定的线程首先告诉 mbarrier 期望多少个字节，然后 CTA 在该 mbarrier 上等待，之后任何 MMA 才能接触 SMEM tile。下图追踪了这一握手的全过程。

![TMA 异步加载：同步流程](../img/tma_sync_flow.png)

上图展示了加载侧的握手：一个选定的线程启动 TMA，mbarrier 计数期望的字节数，MMA 在释放之前等待再读取 SMEM。图中 "Elected Thread" 指的是启动 TMA 的选定线程，在我们的代码中是 `tid == 0` 线程，而不是 `elect_sync()` 通道。

因此，整合加载路径：选定的线程发出两个 `copy_async` 调用，然后跟随 `arrive.expect_tx(total_bytes)`，其中字节数正是 mbarrier 应该等待的数据量。一旦引擎移动了那么多字节，匹配的 `mbarrier.try_wait(phase)` 释放，只有到那时 SMEM tile 才能安全地提供给 MMA。

存储侧走的是相同的硬件，但以不同的方式等待，因此值得在脑中清楚地区分这两个协议：加载通过 mbarrier 和字节数跟踪完成，而存储通过提交组和等待组来跟踪完成。在线程将其 fp16 结果写入 `Dsmem` 并同步之后，一个选定的线程启动 `Tx.copy_async(D[...], Dsmem, dispatch="tma")`，然后 `cp_async.bulk.commit_group()` 后跟 `cp_async.bulk.wait_group(0)` 阻塞，直到存储排空。这个等待不是可选的：在前一个存储完成之前，`Dsmem` 不能被下一个 tile 重用。

**尝试与你的 agent 一起做**：为一个 K tile 跟踪第 4 步的加载和存储同步。确定哪个线程启动每个 TMA 命令，哪个 mbarrier 或提交组跟踪完成，哪个等待保护 MMA 对 `Asmem` 和 `Bsmem` 的读取，以及哪个等待保护 `Dsmem` 的重用。为什么 `elect_sync()` 对于 TMA 加载协议来说是错误的线程选择？

### 完整 Kernel

完整的 kernel 将 TMA 加载和存储嵌入到第 3 步的结构中，保持该结构的其余部分不变。导入与之前相同：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

它被包装在 `hgemm_v4(M, N, K)` 中，这是我们一贯遵循的模式：包装器将依赖于形状的常量和布局与使用它们的 kernel 放在一起。

```python
def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    F16_SIZE = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM allocation (now includes Dsmem for TMA store) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((1,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()
    
        # --- Barrier + TMEM init ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
        if warp_id == 0:
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
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0
    
        # --- Inline helpers ---
        @T.inline
        def tma_load(k_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([0])
            })
            Tx.copy_async(Asmem[:, :],
                          A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[:, :],
                          B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([0]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
            )
    
        @T.inline
        def mma(accum):
            Tx.gemm_async(
                tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                accum=accum, dispatch="tcgen05", cta_group=1
            )
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        # --- K-loop with TMA async ---
        tid = T.meta_var(warp_id * 32 + lane_id)
        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)
    
            # Single thread issues TMA load
            if tid == 0:
                tma_load(k_st)
    
            # Wait for TMA to finish; the mbarrier release carries SMEM
            # visibility to the subsequent MMA, so no extra fence is needed.
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)
    
            # Single thread issues MMA
            if tid == 0:
                mma(accum=k != 0)
    
            # Wait for MMA to finish
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_tma ^= 1
            phase_mma ^= 1
    
        # --- TMA Store Writeback ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
    
        # Read TMEM -> registers (async; wait.ld then cta_sync to ensure read completes)
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        # Cast fp32 -> fp16
        Tx.cast(Dreg_f16[:], Dreg[:])
        # Write registers -> Dsmem, flush, then sync
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        # TMA store: Dsmem -> GMEM. One selected thread starts the store and drains the
        # store group before Dsmem is reused.
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)
    
        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

### Kernel 中的 TMA 配置

该 kernel 中的几乎所有内容都来自第 3 步。只有五个配置点实际承载 TMA 语义，值得逐一了解：

- **TMA 配置**：`{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}` 告诉 `Tx.copy_async` 使用 TMA 并通过 `tma_bar` 报告加载完成。

- **字节计数**：`(BLK_M * BLK_K + BLK_N * BLK_K) * 2` 是两个 fp16 操作数 tile 加载的字节数。`arrive.expect_tx(...)` 将此计数提供给 mbarrier。

- **mbarrier 初始化**：`init(tma_bar.ptr_to([0]), 1)` 创建 TMA 加载使用的完成屏障。

- **`@T.inline`**：`tma_load(...)` 和 `mma(...)` 是辅助函数。它们在编译时展开到 kernel 体中，可以使用外层 kernel 中的变量。

- **TMA 存储同步**：尾声首先将 fp16 行写入 `Dsmem`。`fence.proxy_async` 和 `warpgroup_sync` 使这些线程写入的 SMEM 值准备好供 TMA 存储路径使用。然后存储使用 `commit_group()` 和 `wait_group(0)` 等待 SMEM 到 GMEM 的传输完成。

至此我们有了正确的部件，但节奏不对。第 4 步仍然在每次加载完成后才开始匹配的 MMA，因此加载和乘法从未真正同时运行；我们费了很大力气分离的两个引擎仍然在轮流执行。下一步保持 TMA 加载和存储路径完全不变，而是重新安排调度，使得一个 K tile 的加载可以与另一个 tile 上的计算同时进行。

(chap_software_pipeline)=
## 第 5 步：软件流水线（PIPE_DEPTH=2）

为什么第 4 步不能将加载与计算重叠，而这两个引擎显然是独立的？障碍在于存储。只有一对 SMEM tile 时，下一个加载无处可去：在当前 MMA 完成对该对的读取之前，它无法开始，因为提前开始会覆盖仍在使用的数据。第 5 步通过对共享内存进行双缓冲来消除这个存储冲突。单个 warpgroup 的循环仍然在每次 MMA 之后才启动下一个 TMA 加载，但它现在有了不同的阶段来进行预取和重用。我们仍然在完整的 M=N=K=4096 尺寸上。

> **此步骤更改的内容：布局**
> - 范围：不变，一个 warpgroup。
> - 布局：单个 SMEM tile 对变为一个 `PIPE_DEPTH` 阶段的环形缓冲区。
> - 调度：不变，TMA 加载和 `tcgen05` MMA；此步骤增加了预取和阶段重用，而完整的加载/计算重叠在第 7 步中实现。

### 流水线讲解

使用 `PIPE_DEPTH=2` 时，kernel 分配两个 SMEM 阶段，为加载路径和 MMA 路径提供独立的插槽。

将下图理解为两阶段缓冲区旨在实现的流水线结构，而不是这个单个 warpgroup kernel 的确切执行轨迹。第 5 步构建了环形缓冲区并预取了后续阶段，但主循环仍然在当前的 MMA 完成后才发出下一个 TMA 加载。完整的加载/计算重叠在第 7 步中实现，此时 warp 特化为 TMA 和 MMA 分配了不同的角色。

![*流水线 PIPE_DEPTH=2，目标调度；此单 warpgroup 步骤仅进行预取，完整的重叠将在第 7 步的 warp 特化中实现*](../img/pipe_depth2.png)

一旦准备好了，循环就在两个阶段之间交替。两个 TMA 加载预先填满两个阶段；之后，循环等待当前阶段，对其执行 MMA，等待 MMA 完成对该阶段的读取，然后将 `k + PIPE_DEPTH` 的加载下发到刚刚变得可重用的阶段。这还不是并发的 TMA/MMA 调度，但它建立了环形缓冲区结构，第 7 步将把它拆分为生产者和消费者角色。

具体来说，代码与第 4 步在四个方面有所不同：

1. `Asmem` 和 `Bsmem` 增加了一个前导的 `PIPE_DEPTH` 维度，因此每个阶段都有自己独立的 SMEM 存储。
2. `tma_bar` 变成一个数组，每个阶段有一个 mbarrier。
3. 在主 K 循环之前，kernel 预取前两个阶段。
4. K 循环使用 `stage = k % PIPE_DEPTH`：等待当前阶段，对其执行 MMA，然后重用该阶段用于 `k + PIPE_DEPTH`。

### 流水线机制

**1. 预取**：在主循环运行之前，我们加载前 `PIPE_DEPTH` 个阶段，这样循环在第一次迭代时总能找到等待它的数据：
```python
for s in range(min(PIPE_DEPTH, K_TILES)):
    tma_load(s, s * BLK_K)
```

**2. 主循环**：对于每个 K tile，我们等待其阶段就绪，对其执行 MMA，然后立即将现在空闲的阶段重新投入使用，启动前 `PIPE_DEPTH` 个 tile 的加载：
```python
stage = k % PIPE_DEPTH
wait(tma_bar[stage], phase_tma)
mma(stage, accum)
wait(mma_bar[0], phase_mma)
phase_mma ^= 1
tma_load(stage, next_k * BLK_K)
```

**3. 阶段管理**：这是最容易让人困惑的部分，但规则比它初看起来简单。每个屏障的阶段翻转规则直接取决于该屏障有多少个槽位，这就是为什么两个屏障以不同的节奏翻转。MMA 累加器位于一个 TMEM 槽中，因此 `mma_bar` 是一个每次迭代都会重新访问的单一屏障（`mma_bar.ptr_to([0])`），而每次迭代都会重新访问的屏障必须在每次迭代中翻转其阶段。TMA 屏障则不同：它们形成一个 `PIPE_DEPTH` 元素的数组，每个阶段有一个屏障，而任何给定阶段的屏障只有每轮环形缓冲区才会被再次访问。因此 `phase_tma` 仅在阶段索引回绕到 0 时才翻转：
```python
if stage == PIPE_DEPTH - 1:
    phase_tma ^= 1
```

**尝试与你的 agent 一起做**：使用 `PIPE_DEPTH=2` 和 `K_TILES=5`，让它跟踪主循环。对于每个 `k`，列出 `stage`、传递给等待的 `phase_tma` 和 `phase_mma` 值，以及是否发出了新的预取。`phase_tma` 究竟在哪里翻转，为什么最后两次迭代没有预取？

### 完整 Kernel

完整的 kernel 保持第 4 步的 TMA 加载和存储路径不变，然后将其包裹在我们刚刚描述的分阶段缓冲区和阶段逻辑中。导入不变：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

它被包装在 `hgemm_v5(M, N, K)` 中。`PIPE_DEPTH=2` 常量设置了流水线阶段的数量（这里为两个，正好是双缓冲）：

```python
PIPE_DEPTH = 2

def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    # Double-buffered layouts: first dimension is pipeline stage
    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        # Double-buffered TMA barriers (one per stage), single MMA barrier
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # Initialize barriers: PIPE_DEPTH for TMA, 1 for MMA
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                for s in range(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        @T.inline
        def tma_load(stage, k_offset):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        tid = T.meta_var(warp_id * 32 + lane_id)

        # === Prefetch: load first PIPE_DEPTH stages ===
        if tid == 0:
            for s in range(min(PIPE_DEPTH, K_TILES)):
                tma_load(s, s * BLK_K)

        # === Main loop ===
        for k in range(K_TILES):
            stage = k % PIPE_DEPTH

            # Wait for TMA to finish loading this stage
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

            # MMA on this stage's data
            if tid == 0:
                mma(stage, accum=(k != 0))

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

            # Issue next prefetch load (k + PIPE_DEPTH)
            next_k = k + PIPE_DEPTH
            if next_k < K_TILES:
                if tid == 0:
                    tma_load(stage, next_k * BLK_K)

            # TMA phase flips when stage wraps around
            if stage == PIPE_DEPTH - 1:
                phase_tma ^= 1

        # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

(chap_persistent_kernel)=
## 第 6 步：持久化 Kernel + Tile 调度器

到目前为止，所有优化都针对单个 tile 内部的工作。第 6 步改变了问题的规模，并跨 tile 进行优化。

第 5 步为每个 128 x 128 输出 tile 启动一个 CTA。对于 4096 x 4096 的输出，这意味着 1024 个独立的 CTA，每个都要支付自己的设置成本，然后在其 tile 完成后立即消失。

第 6 步改为启动一个固定数量的 CTA 池，然后让每个 CTA 依次处理多个 tile。这给我们带来两个好处：设置工作在多个 tile 之间分摊，并且 tile 分配移到了 kernel 内部，调度器可以在其中选择重用操作数的顺序。我们保持在完整的 M=N=K=4096 尺寸上。

> **此步骤更改的内容：范围**
> - 范围：一个固定数量的持久化 CTA 池，每个通过调度器循环处理多个输出 tile。
> - 布局：不变，相同的每 tile SMEM/TMEM/寄存器路径。
> - 调度：不变。

### 持久化调度

持久化 kernel 的定义性思想是它根据硬件而非问题来设定其网格大小。它启动 `SM_COUNT` 个 CTA，大约每个 SM 一个，无论有多少个输出 tile，目的是保持每个 SM 持续被占用。我们特意说"大约"：精确的 1:1 驻留并不保证，因为它取决于占用率和硬件选择如何调度 CTA。

在我们这里针对的 B200 上，`SM_COUNT=148`。这 148 个 CTA 中的每一个循环处理由 `ClusterPersistentScheduler2D` 分配给它的 tile。

第一个收益是分摊。TMEM 分配、屏障初始化和调度器状态现在每个 CTA 执行一次，并在该 CTA 处理的大约 7 个 tile 中重复使用，而不是在 1024 个一次性 CTA 中重复 1024 次。

第二个收益来自调度器选择的顺序。设置 `l2_group_size=8` 将附近的 tile 分组在一起，因此共享同一行带的 tile 重用相同的 A 行 tile，而共享同一列带的 tile 重用相同的 B tile。连续执行这些 tile 使操作数在 L2 中保持热状态，而不是从 HBM 重新获取。这正是第 3 步中未利用的重用。

```python
bx = T.cta_id([SM_COUNT])  # 1D grid, one CTA per SM

tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // BLK_M,
    num_n_tiles=N // BLK_N,
    l2_group_size=8,       # Group 8 nearby tiles together
    num_clusters=SM_COUNT
)
tile_scheduler.init(bx)
```

循环处理 tile 带来了一个容易忽略的正确性后果。每个 tile 运行自己全新的 K 循环，这意味着它的屏障阶段必须从已知状态开始。在第 5 步中，一个 CTA 恰好处理一个 tile，因此初始化 `phase_tma` 和 `phase_mma` 一次就完全没问题。在第 6 步中，这些初始化器必须移到 `while tile_scheduler.valid()` 循环*内部*，这样每个 tile 都以与其自身的 TMA 和 MMA 工作匹配的阶段状态开始，而不是继承前一个 tile 可能留下的状态：

```python
while tile_scheduler.valid():
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0
    ...
```

### 完整 Kernel

从结构上看，该 kernel 不过是第 5 步的流水线包裹在一个 tile 级的外层循环中。唯一的新依赖是调度器本身，我们将其与其他导入一起引入：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D
```

网格维度现在只是 `SM_COUNT` 而不是 `(M//BLK_M, N//BLK_N)`，一个 `ClusterPersistentScheduler2D` 接管了为每个 CTA 分配 tile 的工作：

```python
SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
PIPE_DEPTH = 2

def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 1D grid: one CTA per SM (not a 2D grid anymore!)
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation (same as Step 5) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- Barrier + TMEM init (same as Step 5) ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        # Tile scheduler: assigns tiles to CTAs in L2-friendly order
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT
        )
        tile_scheduler.init(bx)

        tid = T.meta_var(warp_id * 32 + lane_id)

        @T.inline
        def tma_load(stage, k_offset, m_st, n_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # === Outer loop: iterate over tiles ===
        while tile_scheduler.valid():
            # Get current tile position from scheduler
            m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

            # === Inner loop: same pipeline as Step 5 ===
            phase_tma: T.int32 = 0
            phase_mma: T.int32 = 0

            # Prefetch first PIPE_DEPTH stages
            if tid == 0:
                for s in range(min(PIPE_DEPTH, K_TILES)):
                    tma_load(s, s * BLK_K, m_st, n_st)

            # Main K-loop
            for k in range(K_TILES):
                stage = k % PIPE_DEPTH
                T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)
                if tid == 0:
                    mma(stage, accum=(k != 0))
                T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma ^= 1
                next_k = k + PIPE_DEPTH
                if next_k < K_TILES:
                    if tid == 0:
                        tma_load(stage, next_k * BLK_K, m_st, n_st)
                if stage == PIPE_DEPTH - 1:
                    phase_tma ^= 1

            # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
            Dreg = T.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = T.alloc_local((BLK_N,), d_type)
            Dreg_wg = Dreg.view(128, BLK_N,
                                layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
            T.ptx.tcgen05.wait.ld()
            T.cuda.cta_sync()
            Tx.cast(Dreg_f16[:], Dreg[:])
            Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.warpgroup_sync(10)
            if tid == 0:
                Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                              Dsmem[:, :], dispatch="tma")
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(10)

            T.cuda.cta_sync()
            tile_scheduler.next_tile()  # Move to next tile

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## 练习

1. 在第 4 步中，`arrive.expect_tx` 使用了 `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` 字节。如果这个字节数太小或太大，mbarrier 会等待什么？
2. 在第 5 步中，为什么每个 SMEM 阶段需要自己的 TMA 屏障，而不是为两个阶段共享一个 `tma_bar`？
3. 在第 6 步中，一个 4096 x 4096 的输出，使用 `BLK_M=BLK_N=128`，有多少个输出 tile？使用 `SM_COUNT=148`，每个持久化 CTA 平均处理多少个 tile？
