(chap_gemm_advanced)=
# 通过 Warp Specialization 和 Cluster 扩展 GEMM

:::{admonition} 概览
:class: overview

- 流水线 GEMM 仍然让一个 warpgroup 顺序执行加载、MMA 和写回，这是本章要消除的瓶颈。
- 第 7 步将 warp 特化为不同角色，第 8 步添加 2-CTA cluster，第 9 步添加多个消费者。
- 每一步消除一个串行瓶颈，最终接近当前最优吞吐量。
:::

上一章的流水线 GEMM（{ref}`chap_gemm_async`）速度很快，但它仍然要求一个 warpgroup 完成所有事情：发起加载、运行 MMA、然后将结果写回。即使有了软件流水线，这组线程仍然是三种引擎交汇的地方。

症状很容易看到。当 Tensor Core 运行时 TMA 单元空闲，当结果写回内存时 Tensor Core 空闲，每种引擎都通过一组线程等待其他引擎。突破的方法是不要再让一个团队做所有事情。

我们通过三个扩大协作的步骤来实现这一想法。第 7 步（{ref}`chap_warp_specialization`）将 warp 特化为生产者、消费者和写回三种角色。第 8 步（{ref}`chap_cta_cluster`）将两个 CTA 组成一个 cluster，共享它们的共享内存中的操作数。第 9 步（{ref}`chap_multi_consumer`）添加第二个 MMA 消费者，使得一个分阶段 tile 能支撑两倍的数学运算。

将这三个步骤视为同一模式在不同尺度上的应用会很有帮助。第 7 步将完整的流水线保持在一个 CTA 内：TMA 和 MMA 共享一个 warpgroup，而写回在另一个中运行。第 8 步将协作扩展到跨 CTA，产生一个横跨两者的 256×256 tile。第 9 步进一步推动计算密度：cluster 输出增长到 512×256，每个分阶段的 B tile 被两个消费者复用，我们达到了本教程中最密集的变体。

在这一切中，有一件事保持不变。SMEM、TMEM 和寄存器布局仍然遵守我们在前两章中建立的契约；变化的是*谁在协作*，而不是数据的布局方式。第 8 步是协作范围首次超出一个 CTA，因此其操作数 tile 分布在两个 CTA 的共享内存中，一个布局沿着 `cbx` cluster 轴跨越两个 CTA。


(chap_warp_specialization)=
## 第 7 步：Warp Specialization + 流水线

单 warpgroup kernel 在性能上有所损失，原因很简单：每个线程走相同的路径，加载、计算、写回依次进行，所以当它在加载时，Tensor Core 无事可做，当它在计算时，TMA 引擎也无事可做。修复方法是 *warp specialization*。我们不再让一组线程依次完成每项工作，而是将每项工作分配给一个专用的 warp，让这些 warp 同时运行，通过软件流水线将它们串联起来。这是 GEMM 路径中最大的架构变化，本章的其余部分都建立在此基础上。此处的基准测试使用 M=N=K=4096。

> **本步的变化：范围**
> - 范围：一个 warpgroup 顺序执行加载→MMA→写回，变为三个并发角色（TMA 生产者、MMA 消费者、写回），通过 full/empty barrier 连接。
> - 布局：不变，与第 6 步相同的 SMEM 阶段和 TMEM 累加器。
> - 分发：不变，TMA 加载，`tcgen05` MMA。

**主题。**

- Warp specialization：将不同的 warp/warpgroup 专用于不同的任务

- 高级 barrier 抽象：`TMABar`、`TCGen05Bar`、`MBarrier`

- `PipelineState` 用于自动阶段/相位管理

- `warpgroup_sync` barrier ID 用于按 warpgroup 同步

（多阶段 SMEM 流水线和持久化的 `ClusterPersistentScheduler2D` 从第 5-6 步原样复用；此处只有范围拆分是新的。）

### 从串行到并发

在介绍角色和 barrier 之前，我们先隔离出 warp specialization 要消除的调度瓶颈。下图使用第 4 步风格的串行时间线作为第 4-6 步特化前 kernel 的紧凑参考，然后将其放在第 7 步 warp 特化调度之上，以便一目了然地看到引擎利用率上的差异。

![Warp Specialization 时间线](../img/warp_specialization_timeline.png)

上方是特化前的单 warpgroup 模式：同一组未特化的线程同时拥有加载路径和 MMA 路径，因此当一个引擎活跃时另一个引擎很容易空闲。第 5 步和第 6 步通过双缓冲和持久化调度改善了该基线，但它们尚未将加载和计算拆分为独立的生产者和消费者角色。下方，特化打破了这种轮替。TMA 生产者在 MMA 消费者忙于计算时预取下一个 tile，而写回独立进行。生产者 warp 3 在消费者 warp 0 仍在处理当前 MMA 时发起下一个加载，因此两者都不必等待对方。加载/MMA 握手使用两个 barrier：

- **`tma2mma`**（TMA → MMA）：表示已加载的 SMEM 数据已准备好供 MMA 消费。
- **`mma2tma`**（MMA → TMA）：表示 MMA 已完成读取一个缓冲区，因此 TMA 可以将其重用于下一个加载。

图中有一个细节乍看可能像错误：`mma2tma` 箭头跳过一个阶段。原因是环形缓冲区。当 `PIPE_DEPTH=2` 时有两个 SMEM 缓冲区，阶段 0 和阶段 1；TMA Load k=0 填充缓冲区 0，TMA Load k=1 填充缓冲区 1。当 MMA Compute k=0 完成读取缓冲区 0 时，它发出 `mma2tma` 信号表示该缓冲区已空闲，但实际需要回收缓冲区 0 的加载是 TMA Load k=2，而不是 k=1（k=1 正在使用缓冲区 1）。这就是为什么来自 MMA Compute k=0 的 `mma2tma` 箭头一直延伸到 TMA Load k=2。释放跳过一个阶段，仅仅是因为环有两个槽位。

### Warp 角色

时间线展示了*为什么*要拆分工作；下一个问题是*谁*来做每个部分。特化将三个工作（加载、计算、写回）分配给特定的 warp，以便它们可以同时运行。当 `WG_NUMBER=2` 时，kernel 使用两个 warpgroup（在角色表中缩写为 WG）：

| 角色 | 位置 | 工作 |
|------|----------|-----|
| **TMA 生产者** | Warpgroup 1, warp 3 | 通过 TMA 持续加载 A 和 B tile |
| **MMA 消费者** | Warpgroup 1, warp 0 | 数据就绪后立即运行 MMA |
| **写回** | Warpgroup 0（所有 warp） | 读取 TMEM 结果，写入 GMEM |

### 4 个 Barrier

三个并发角色需要四个 barrier，这四者整齐地分为两个相反的方向。前向路径（TMA → MMA → 写回）表示数据*就绪*；其消息是"你等待的 tile 已到达"。后向路径（写回 → MMA → TMA）表示缓冲区*释放*："你想要的槽位又空闲了"。一旦你了解命名约定，名称就不言自明：每个都是 `source2destination`，所以 `tma2mma` 就是 TMA 用来通知 MMA 的 barrier。

| Barrier | 类型 | 方向 | 含义 |
|---------|------|-----------|---------|
| **tma2mma** | `TMABar` | TMA -> MMA | "SMEM 数据已就绪" |
| **mma2tma** | `TCGen05Bar` | MMA -> TMA | "SMEM 缓冲区可以重用" |
| **mma2ld** | `TCGen05Bar` | MMA -> 写回 | "TMEM 结果已就绪" |
| **ld2mma** | `MBarrier` | 写回 -> MMA | "TMEM 可用于下一个 tile" |

为什么每个 barrier 具有其特定的*类型*？类型取决于生产者如何宣布它已完成。**TMA 加载**使用 `TMABar`，这是一种带有字节计数的 mbarrier：TMA 硬件本身在传输的字节到达后到达 barrier，因此消费者无需任何线程轮询即可知道数据已就绪。**TMA 存储**不能使用这种方式（存储没有通知对象），所以它们回退到 `cp_async.bulk.commit_group()` + `wait_group(0)`，其中发起线程简单地等待自己的写入完成。**MMA 操作**使用 `TCGen05Bar`，其中 `tcgen05.commit()` 指令在 MMA 完成时通知 barrier。

这里有一个小细节在第 8 步中会得到回报。`arrive` 调用传递 `cta_mask=0`，因为在单 CTA kernel 中没有其他 CTA 需要通知。当第 8 步形成 cluster 时，这个参数会变为非零，成为唤醒协作 CTA 的机制。

### PipelineState

四个 barrier 告诉各角色缓冲区*何时*就绪；但还需要有东西来跟踪流水线循环时每个角色处于*哪个*缓冲区。这个簿记工作由 `PipelineState` 管理。环形缓冲区同时携带两部分簿记：我们当前在哪个槽位，以及我们正在等待该槽位 barrier 的哪个"相位"。在流水线循环中手动跟踪这两者是滋生 off-by-one 错误的典型来源，而一个 off-by-one 错误会导致整个 kernel 死锁。`PipelineState` 的存在就是为了让你不必手动维护这两者：

```python
tma_ps = PipelineState(PIPE_DEPTH, phase=1)   # 生产者开始时准备就绪（phase=1）
# tma_ps.stage = 当前阶段索引
# tma_ps.phase = 当前相位（0 或 1）
tma_ps.advance()                          # 前进到下一个阶段
```

初始 `phase` 决定一个角色的第一个 `wait` 是让它运行还是阻塞，正确的答案在流水线的两端是相反的，这正是容易出错的地方：
- `phase=1`（生产者）-> 第一个 `wait(phase=1)` 看到 barrier 仍处于 phase 0，由于 0 != 1，它**立即通过**。这正是我们想要的，因为缓冲区初始是空的，生产者应该能够毫不延迟地开始填充它们。

- `phase=0`（消费者）-> 第一个 `wait(phase=0)` 看到 barrier 处于 phase 0，由于 0 == 0，它**阻塞**。同样正是我们想要的，因为还没有数据，在生产者到达之前消费者没有东西可读。

如果两端使用相同的起始 phase，你会得到死锁，或者更糟糕的是，静默数据损坏，所以这一个选择值得做对。

### `warpgroup_sync` Barrier ID

特化引入了一种容易踩入的同步危险。一旦每个 warpgroup 运行不同的代码路径，熟悉的 `cta_sync()` 会死锁：它使用硬件 barrier #0 并要求*每个*CTA 线程到达，然而在 warpgroup 分支内部只有部分线程存在。我们需要的是一个范围限定在单个 warpgroup 的 barrier。GPU 提供 16 个命名 barrier（ID 0-15），所以 kernel 使用 `warpgroup_sync(10)`，它只同步一个 warpgroup 内的线程。当多个 warpgroup 各自需要自行同步时（如第 9 步的多消费者场景），它们通过 `warpgroup_sync(wg_id + 10)` 使用不同的 ID，以确保它们永远不会在同一个硬件 barrier 上冲突。

**实现。**

这里我们使用 `PIPE_DEPTH=2`，这是仍然能实现加载和计算重叠的最小深度。更深的流水线可以隐藏更多的内存延迟，直到 SMEM 预算上限；下文*第 7 步出问题时*的讨论详细分析了这种权衡。现在有了所有组件（角色、四个 barrier、`PipelineState` 和 warpgroup 范围的同步），我们可以组合出完整的 kernel：

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.pipeline import TMABar, TCGen05Bar, MBarrier, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D

SM_COUNT = 148  # NVIDIA B200 GPU 上的 SM 数量
F16_SIZE = 2

def hgemm_v7(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- 分配 ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

        # --- Barrier 初始化 ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128)   # 所有 128 个 Warpgroup 0 线程到达
        pool.commit()

        # --- TMEM 分配 + fence ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile 调度器 ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)
        tile_scheduler.init(bx)
        m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

        # =============================================
        # Warpgroup 1: TMA 生产者（warp 3）+ MMA 消费者（warp 0）
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                # === TMA 生产者 ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            tma2mma.arrive(tma_ps.stage,
                                           (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                # === MMA 消费者 ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        # 等待 TMEM 从前一个 tile 的写回中释放
                        ld2mma.wait(ld_ps.stage, ld_ps.phase)
                        ld_ps.advance()

                        for k in range(K_TILES):
                            tma2mma.wait(mma_ps.stage, mma_ps.phase)
                            Tx.gemm_async(
                                tmem[:, :BLK_N],
                                Asmem[mma_ps.stage, :, :],
                                Bsmem[mma_ps.stage, :, :],
                                accum=(k != 0), dispatch="tcgen05", cta_group=1)
                            mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
                            mma_ps.advance()

                        # 通知结果已准备好用于写回
                        mma2ld.arrive(0, cta_group=1, cta_mask=0)
                        tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: 写回
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((BLK_N,), d_type)

            while tile_scheduler.valid():
                # 等待 MMA 结果
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()

                # 读取 TMEM -> 寄存器（warpgroup 范围）
                reg = T.alloc_local((BLK_N,), acc_type)
                reg_wg = reg.view(128, BLK_N,
                    layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
                Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
                T.ptx.tcgen05.wait.ld()

                # 通知 TMEM 空闲（所有 128 个线程到达）
                ld2mma.arrive(0, cta_id=0, pred=True)

                # 转换 fp32 -> fp16
                Tx.cast(reg_f16[:], reg[:])

                # 写入 Dsmem + TMA 存储
                Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)
                if warp_id == 0:
                    if lane_id == 0:
                        Tx.copy_async(D[m_st:m_st+BLK_M, n_st:n_st+BLK_N],
                                      Dsmem[:, :], dispatch="tma")
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(10)

                tile_scheduler.next_tile()

        # --- 清理 ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

要运行这些 kernel，复用我们在第 1 步（{ref}`chap_gemm_basics`）中展示过一次的编译/运行/检查框架：将 `hgemm_v1` 替换为 `hgemm_v7`、`hgemm_v8` 或 `hgemm_v9`，并选择一个问题规模，如 `M=N=K=4096`。请注意，cluster 步骤要求 `M` 和 `N` 是其 cluster tile 的倍数（第 8 步为 `256×256`，第 9 步为 `512×256`），因此极小的 `128×128` 规模根本不会产生任何 tile。每个步骤在独立的 Python 会话中编译，切换步骤前重启 kernel，因为这些 kernel 复用内部名称，编译器会保留每个会话的状态。各步骤的计时汇总在*端到端结果*中。

### Epilogue（写回）详情

第 7 步可以使用一个相当简单的 epilogue。只有 `BLK_N=128` 列，写回 warpgroup 一次性将整个 TMEM tile 读入寄存器，然后发起一次 TMA 存储。第 8 步和第 9 步将没有这种便利，这正是它们后来引入块处理的原因，但现在流程是：

1. 等待 MMA：`mma2ld.wait(phase)`。本教程中的第 8 步和第 9 步在此处添加了一个保守的 `fence.after_thread_sync()`；MMA 完成 mbarrier 已经覆盖了排序，大多数 kernel（包括 CUTLASS）省略它，因此第 7 步也省略了。
2. 读取 TMEM -> 寄存器（每个线程 128 个 fp32，warpgroup 范围通过 `Tx.copy_async(reg_wg, tmem[:, :BLK_N])` 后跟 `T.ptx.tcgen05.wait.ld()`）。
3. 通知 MMA：`ld2mma.arrive(0, cta_id=0, pred=True)`（所有 128 个线程到达）；TMEM 现在可用于下一个 tile。这两个 `arrive` 关键字在 cluster 步骤中会反复出现：`cta_id` 指定要通知*哪个 CTA 的*barrier 副本（`0` = 此 CTA，本地 barrier；在第 8 步中协作 arrive 通过 `cta_mask` 指向 CTA-0），而 `pred` 是每个线程的谓词，控制该线程是否实际到达（这里是 `True`，所以每个写回线程都计入到达总数）。
4. 在寄存器中将 fp32 转换为 fp16。
5. 写入寄存器 -> Dsmem，然后 `fence.proxy_async("shared::cta") + warpgroup_sync(10)` 进行刷新。
6. TMA 存储 Dsmem -> GMEM，通过 `cp_async.bulk.commit_group() + wait_group(0)`。

第 8 步（`BLK_N=256`）和第 9 步（每个消费者 `MMA_N=256`）不能保持这种单次遍历的形式，原因是寄存器压力。每个线程读取 256 个 fp32 值意味着 256 × 4 = 1024 字节必须同时驻留在每个线程的寄存器中，这有可能溢出到 local memory，并且还强制要求更大的 Dsmem 缓冲区。因此这些步骤将写回分解为 `EPI_N` 列的块（`EPI_N=64`）：每次迭代只保持 `EPI_N` 个 fp32 寄存器活跃，并发出相应较小的 TMA 存储，用几个额外的存储指令换取保持舒适的寄存器预算。

**实现说明。**

- **持久化 kernel**：`bx = T.cta_id([SM_COUNT])` — 每个 SM 一个 CTA，循环处理 tile

- **L2 友好的调度**：`ClusterPersistentScheduler2D` 为缓存局部性排序 tile

- 这种模式——warp specialization 加软件流水线——在包括 CUTLASS 风格设计在内的高性能 GEMM kernel 中很常见。

### 第 7 步出问题时

第 7 步是第一个 TMA 加载、`tcgen05` MMA 和写回同时进行的 GEMM kernel。相同的失败模式在第 8 步和第 9 步中会再次出现：barrier 计数不匹配、角色守卫位置错误、缺少 fence，或在 TMA 存储完成前重用暂存缓冲区。这些情况的调试清单收集在 {ref}`chap_warp_spec_debug` 中。

**流水线深度调优。** 第 7 步 kernel 以 `PIPE_DEPTH=2`（最小值）运行。将其推到 4 或 6 可以让 TMA 生产者领先 MMA 消费者更远、隐藏更多内存延迟，但代价是消耗更多的 SMEM，而 SMEM 是有限的。B200 每个 SM 提供 228 KB（参见 {ref}`chap_background` 中的*需要记住的数字*）。当 `BLK_M=BLK_N=128, BLK_K=64, fp16` 时，每个流水线阶段 A 和 B 共消耗 `(128*64 + 128*64) * 2 = 32 KB`，而 `Dsmem` 写回暂存缓冲区再增加 32 KB。这意味着 `PIPE_DEPTH=4` 大约需要 160 KB，`PIPE_DEPTH=6` 大约需要 224 KB，刚好到达预算上限。要超过这个深度，必须重新考虑写回暂存策略。

---

Warp specialization 使一个 CTA 的线程实现了协作。下一步将这种协作扩展到 CTA 的边界之外，让两个 CTA 共同处理一个更大的 tile。


(chap_cta_cluster)=
## 第 8 步：2-CTA Cluster

第 7 步让引擎实现了重叠，但每个 CTA 仍然孤立地计算自己的 128×128 tile，重复加载相邻 CTA 无法借用的操作数。第 8 步打破了这种隔离。两个 CTA 组成一个 cluster，获得了访问彼此共享内存的能力，因此一次协作的 `tcgen05` MMA 产生一个横跨两者的 256×256 tile，而一次 B 的加载现在可支撑两倍的 MMA 工作。与之前一样，M=N=K=4096。

> **本步的变化：范围 + 布局 + 分发**
> - 范围：协作范围现在横跨 cluster 中的两个 CTA，而不是一个。
> - 布局：操作数 tile 分布在两个 CTA 的 SMEM 中；CTA 0 拥有共享完成 barrier（`remote_view`）。
> - 分发：MMA 获得了 `cta_group` / `cta_mask`，使得 `tcgen05` 作为 2-CTA 协作操作运行。

**主题。**

- CTA cluster：多个 CTA 协作处理更大的 tile

- 通过 `map_shared_rank` 进行跨 CTA SMEM 访问

- `cta_group=2` 用于在 256x256 cluster tile 上进行协作 MMA

- 使用 `cta_mask` 进行跨 CTA barrier 通知


### Cluster Tile 形状

整个优化建立在一个硬件能力之上：当 `cta_group=2` 时，允许 MMA 读取由*两个* CTA（而不仅仅是它所在的 CTA）暂存的操作数 tile。每个 CTA 加载一个 128 行的存储 B 切片，转置后变成 128 个逻辑输出列，协作 MMA 将两个切片拼接回一个操作数。下图追踪了两个 CTA 的 A 和 B 切片如何组合成单个 256×256 cluster tile：

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="2-CTA cluster：通过跨 CTA SMEM 读取实现协作 MMA" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互式：每个 CTA 拥有一个 A 行切片和一个存储 B 行切片，然后跨 cluster（DSMEM）读取另一个 CTA 的存储 B 切片。`B.T` 之后，两个存储 B 切片覆盖完整的输出列范围，因此这对 CTA 产生一个 256×256 输出 tile。*

**为什么 A 和 B 在 cluster 中拆分**：要了解 256×256 tile 是如何划分的，请回忆本教程将 GEMM 存储为 `D = A @ B.T`，其中存储的 B 的形状为 `N x K`。cluster 中有两个 CTA，拆分会很自然地得出：

- **A 垂直拆分**：CTA-0 持有 A0（行 0-127），CTA-1 持有 A1（行 128-255）。堆叠后：`[A0; A1]`（256 行）。
- **存储 B 按行拆分**：CTA-0 加载 B 行 0-127，CTA-1 加载 B 行 128-255。因为数学运算使用 `B.T`，这两个存储行切片变成逻辑右操作数的两个 128 列切片。
- 当 `cta_group=2` 时，MMA 硬件通过跨 CTA 共享内存访问从**两个** CTA 的 SMEM 读取 B，因此它看到完整的逻辑输出列范围。
- 结果：两个 CTA 协作处理一个 256x256 输出 tile。每个 CTA 写入该 tile 的一个 128x256 行带。

值得停下来思考为什么这是真正的收益而不仅仅是工作的重新洗牌。每个 CTA 仍然只加载 128×K 的 A 和 128×K 的 B，因此 cluster 整体暂存的操作数约为一个 CTA 的两倍，但它产出一个 256×256 tile，携带约 4 倍于 128×128 tile 的输出 FLOPs。因此 MMA 对每个暂存操作数字节做了大约两倍的工作，因为每个 CTA 的 B 切片通过协作 MMA 与另一个 CTA 的 A 切片一起被重用。换句话说，算术强度大约翻倍了，而这正是一个仍偏向内存的 kernel 所需要的杠杆：端到端表中约 2.2× 的加速来自于将相同的字节喂给更多的数学运算。

### Tile 地址计算

现在 cluster 是工作单元，tile 调度器也必须以 cluster tile 为单位计数。它返回的每个 `(m_idx, n_idx)` 命名一个完整的 256×256 区域，cluster 内的两个 CTA 在该区域之间进行拆分。将 cluster 坐标转换为每个 CTA 实际加载的切片，看起来是这样的：

```python
m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
```

两个 CTA 处理*相同的* 256×256 cluster tile，唯一的坐标 `cbx`（CTA 在 cluster 中的位置，0 或 1）选取该 CTA 在两个轴上的贡献。`m_st` 选择此 CTA 拥有的输出行带，`n_st` 选择它馈入协作 MMA 的存储 B 切片，写回随后发出 256 列输出范围的两个 128 列半部分。还要注意 `num_m_tiles = M // 256` 和 `num_n_tiles = N // 256` 计数的是 cluster tile，而非单个 CTA tile。

乍看之下 `cbx` 同时出现在 `m_st` 和 `n_st` 中，就好像行偏移不知何故泄漏到了列中，但这两处使用都是正确的，值得厘清原因。在写回路径上，`cbx` 仅属于 M 轴：每个 CTA 拥有一个不同的 128 行带（`m_st = (m_idx * CTA_GROUP + cbx) * BLK_M`，所以 CTA-0 写入行 `m_idx*256 .. +128`，CTA-1 写入接下来的 128 行），然而两个 CTA 都写入 cluster tile 的*完整* 256 个输出列。这就是为什么存储从 cluster 的 `n_idx` 导出其列（`n_st_epi = n_idx * 256 + no * 128`，看不到 `cbx`）而非从每个 CTA 的 `n_st` 导出。`n_st` 携带 `cbx` 的原因是每个 CTA 加载不同的存储 B 行切片到 MMA 中：在那里，`cbx` 是一个*加载*偏移，而不是 CTA 的输出列偏移。

### 相对于第 7 步的代码更改

与第 7 步的 diff 有六处编辑，每处编码了我们刚刚描述的 cluster 契约的一个部分：

```python
# 1. Cluster 启动
cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])   # cbx = cluster 内 CTA 索引（0 或 1）

# 2. 协作 MMA（原来是 cta_group=1）
Tx.gemm_async(..., cta_group=2)

# 3. 跨 CTA 共享内存访问
B_remote = T.ptx.map_shared_rank(Bsmem, cta_id=1)

# 4. 跨 CTA barrier
tma2mma_cta0 = T.decl_buffer(
    [CTA_GROUP], "uint64",
    data=T.ptx.map_shared_rank(tma2mma.ptr_to([0]), 0),
    scope="shared"
)

# 5. mma2tma / mma2ld arrive 从 cta_mask=0（单 CTA，第 7 步）
#    变为 cta_mask=3（通知 cluster 中的两个 CTA）
mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)

# 6. Cluster sync 在末尾替换 cta_sync
T.cuda.cluster_sync()
```


### Cluster 范围变更

这六处编辑都源于同一个转变：协作范围现在是 cluster 而非单个 CTA。以下几点详细说明了这种扩展在实践中的意义：每个 CTA 如何找到自己的位置，cluster 以谁的 barrier 为协调点，以及哪个 CTA 实际发起协作 MMA。

- **Cluster CTA ID**：`cbx` 告诉每个 CTA 它在 cluster 中的位置（0 或 1）。CTA-0 处理 A 行 0-127，CTA-1 处理行 128-255。

- **远程 barrier 视图**：在 cluster 中，每个 CTA 有自己的 SMEM 和自己的 barrier，这引发了一个明显的问题：如果 CTA-1 需要等待 CTA-0 产生的东西，它实际触碰的是谁的 barrier？答案是提名 CTA-0 的 barrier 作为唯一的协调点，让 cluster 中的任何 CTA 都能访问它们。`map_shared_rank(tma2mma.ptr_to([0]), 0)` 返回指向 CTA-0 barrier 的 cluster 范围指针，TIRx 包装为 `tma2mma.remote_view(0)`，此后每次 arrive 和 wait 都针对 CTA-0 的副本。

- **MMA 仅从 CTA-0 发起**：人们很容易将 `cta_group=2` 理解为并行启动两个引擎，但实际情况并非如此。CTA-0 发出恰好一个 `tcgen05.mma`，硬件随后驱动一个*单一的协作* MMA，跨两个 CTA 读取操作数（来自两个 SM 的 SMEM），并将累加器写入两个 SM 的 TMEM。CTA-1 根本不发出 MMA。（每个 SM 只有一个 `tcgen05` 引擎，所以 `cta_group=2` 是一个跨 SM 的 MMA，而不是两个引擎并行运行。）这就是为什么代码用 `if cbx == 0:` 守卫 MMA。

- **多播 arrive**：`tcgen05.commit(..., cta_group=2, cta_mask=3)` 仅由 CTA-0 发出，但通知两个 CTA 的 barrier。`cta_mask=3`（二进制 `11`）意味着 CTA-0 和 CTA-1 都是目标。

- **ld2mma 初始计数**：`init(128 * CTA_GROUP)` — 两个 CTA 的写回 warpgroup（每个 128 线程）都到达。


**实现。**

```python
def hgemm_v8(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    WG_NUMBER = 2
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, 128))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- 分配 ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, 1)
        ld2mma  = MBarrier(pool, 1)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, 128), d_type, layout=D_layout)

        # --- Barrier 初始化 ---
        tma2mma.init(1)
        mma2tma.init(1)
        mma2ld.init(1)
        ld2mma.init(128 * CTA_GROUP)  # 两个 CTA 的写回线程
        pool.commit()

        # --- TMEM 分配（协作） ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile 调度器（cluster tile） ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        # --- 跨 CTA barrier 视图 ---
        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 1: TMA 生产者（warp 3）+ MMA 消费者（warp 0）
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(ld_ps.stage, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, :MMA_N],
                                    Asmem[mma_ps.stage, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0: 写回（256 列，分 2 个 128 列块）
        # =============================================
        elif wg_id == 0:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((128,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                for no in T.unroll(2):  # 2 个 128 列块 = 共 256 列
                    reg = T.alloc_local((128,), acc_type)
                    reg_wg = reg.view(128, 128,
                        layout=TileLayout(S[(128, 128) : (1@tid_in_wg, 1)]))
                    Tx.wg.copy_async(reg_wg[:], tmem[:, no * 128:(no + 1) * 128])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(10)
                    if warp_id == 0:
                        if lane_id == 0:
                            n_st_epi = T.meta_var(n_idx * 256 + no * 128)
                            Tx.copy_async(D[m_st:m_st+BLK_M, n_st_epi:n_st_epi+128],
                                          Dsmem[:, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)

                ld2mma.arrive(0, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- 清理 ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**2 CTA 的变化。**

- `CTA_GROUP = 2`，`MMA_N = BLK_N * CTA_GROUP = 256`

- `ld2mma.init(128 * CTA_GROUP)` — 两个 CTA 的写回 WG 都到达

- TMA arrive 字节计数包括两个 CTA：`CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`

- `tcgen05.alloc` 和 `tcgen05.dealloc` 必须使用 `cta_group=2`

- 写回将 256 个输出列拆分为两个 128 列块——一次性读取所有 256 个 TMEM 列超过了寄存器容量。第 9 步将块进一步缩小到 `EPI_N=64`

- 末尾用 `cluster_sync()` 替换 `cta_sync()`（确保所有 CTA 完成后才释放 TMEM）

所有这些额外的算术强度直接反映在时钟时间上：第 8 步在 4096³ 规模上达到 **0.104 ms**，比相同规模的第 1 步算法（70 ms）快约 676×（参见端到端表）。Kernel 现在正朝着计算绑定的方向发展，而这正是为第 9 步做的准备，在第 9 步中我们将添加第二个 MMA 消费者以保持更多 Tensor Core 工作在进行中。

如果第 8 步比第 7 步*更慢*，罪魁祸首几乎总是某个新引入的 cluster 契约中的细微错误。有三件事值得首先检查：TMA arrive 字节计数是 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`；调度器维度是 `num_m_tiles=M//256, num_n_tiles=N//256`（对应 256×256 cluster tile）；写回发出两次 TMA 存储，每个 128 列块一次，每次存储都在 Dsmem 被重用之前完成。

---

Cluster 提升了*跨* CTA 的重用。最后一步转向内部，通过为生产者添加第二个 MMA 消费者来保持运算密度，提升*每个* CTA 的计算密度。


(chap_multi_consumer)=
## 第 9 步：多消费者 Warp Specialization

到第 8 步时 MMA 已经是真正忙碌的了，但单个消费者 warp 处理分阶段的 B tile 的速度有限，而 B tile 全程都放在 SMEM 里，任何愿意读取它的 warp 都可以访问。最终的优化正是利用这一点：添加第二个 MMA 消费者，将*不同的* A block 与*相同的* B tile 相乘。每个 CTA 的计算密度翻倍，cluster 输出从 256×256 增长到 512×256。与之前一样，M=N=K=4096。

> **本步的变化：范围 + 布局**
> - 范围：一个 MMA 消费者变为两个，由 `warp_id` 选择。
> - 布局：一个分阶段的 B tile 被两个消费者复用；A 增加了消费者轴。
> - 分发：不变。

**主题。**

- 多个 MMA warp（消费者）以获得更高吞吐量

- 多个写回 warpgroup 与独立的 barrier 槽位

- 本教程中最优化的 GEMM 变体所使用的结构


### 多消费者结构

添加第二个消费者意味着 kernel 现在有更多不同的角色需要布局：两个 MMA warp 而非一个，以及一个匹配的第二个写回 warpgroup 来排空额外的累加器。当 `NUM_CONSUMER=2` 和 `WG_NUMBER=3` 时，kernel 现在横跨三个 warpgroup（在角色表中缩写为 WG）：

| Warpgroup | Warp | 角色 |
|-----------|------|------|
| **WG 2** | warp 0 | MMA 消费者 0：`Asmem[..., 0] x B` -> TMEM 列 `[0:256]` |
| **WG 2** | warp 1 | MMA 消费者 1：`Asmem[..., 1] x B` -> TMEM 列 `[256:512]` |
| **WG 2** | warp 3 | TMA 生产者：每个阶段加载 2 个 A block + 1 个 B block |
| **WG 0** | all | 消费者 0 的写回：读取 TMEM `[0:256]` |
| **WG 1** | all | 消费者 1 的写回：读取 TMEM `[256:512]` |

整个安排取决于一个不对称性。每个消费者将自己的 A block 与*相同的*分阶段 B tile 相乘，因此一次 B 加载现在支撑 2 倍的 MMA 工作，B 的每次有用 FLOP 的加载成本实际上减半了。我们共享 B 而不是 A 的原因是，两个消费者覆盖不同的 M 行带：它们的 A block 是真正不同的数据，而 B 对两者是相同的。练习 3 要求你说服自己这是唯一可行的共享方式。

### 相对于第 8 步的更改

具体而言，支持第二个消费者在几个地方触动了 kernel，每个更改都追溯到一个事实：现在每个阶段有两个 A block 和两个 TMEM 范围需要输入和排空，而 B 保持共享。下面的编辑为每个阶段暂存一个额外的 A block，为每个消费者提供自己的 barrier 槽位，并为更高的 512×256 cluster tile 调整 tile 地址。

- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` — 每个阶段 2 个 A block，每个消费者一个

- TMA 加载 `Asmem[stage, 0]` 和 `Asmem[stage, 1]`，TMA arrive 字节数现在为 `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`（额外的 A block）

- MMA warp 的 `warp_id` 选择哪个 A block 和 TMEM 范围

- `mma2tma.init(NUM_CONSUMER)` — 两个消费者每个阶段都通知 TMA

- `mma2ld` 和 `ld2mma` 具有 `depth=NUM_CONSUMER` — 每个消费者使用自己的 barrier 槽位（MMA 侧用 `warp_id`，写回侧用 `wg_id`）

- Tile 地址：`m_st = (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M` — M 方向有额外的 `NUM_CONSUMER` 因子，因为每个 cluster tile 现在在 M 方向横跨 `NUM_CONSUMER` 个消费者。Tile 调度器使用 `num_m_tiles = M // 256 // NUM_CONSUMER`（cluster tile 是 512x256）

- 写回使用块化的 `EPI_N`，使得每次迭代在寄存器中保留更少的 TMEM 回读值


**实现。**

```python
def hgemm_v9(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N * CTA_GROUP   # 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    WG_NUMBER = 3
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (NUM_CONSUMER, BLK_M, EPI_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- 分配 ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma2mma = TMABar(pool, PIPE_DEPTH)
        mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
        mma2ld  = TCGen05Bar(pool, NUM_CONSUMER)   # depth=2, 每个消费者一个槽位
        ld2mma  = MBarrier(pool, NUM_CONSUMER)     # depth=2, 每个消费者一个槽位
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)

        # --- Barrier 初始化 ---
        tma2mma.init(1)
        mma2tma.init(NUM_CONSUMER)  # 每个阶段预期 2 次到达
        mma2ld.init(1)              # 每个槽位获得 1 次到达
        ld2mma.init(128 * CTA_GROUP)  # 两个 CTA 的写回线程
        pool.commit()

        # --- TMEM 分配（协作） ---
        if wg_id == 0:
            if warp_id == 0:
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile 调度器（512x256 cluster tile） ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256 // NUM_CONSUMER, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 2: TMA 生产者（warp 3）+ 2 个 MMA 消费者（warp 0, 1）
        # =============================================
        if wg_id == 2:
            if warp_id == 3:
                # === TMA 生产者：每个阶段加载 2 个 A block + 1 个 B block ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    m_st_c1 = T.meta_var(m_st + CTA_GROUP * BLK_M)
                    Tx.copy_async(Asmem[tma_ps.stage, 0, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Asmem[tma_ps.stage, 1, :, :],
                                  A[m_st_c1:m_st_c1+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id < NUM_CONSUMER:
                # === MMA 消费者：warp_id 选择 A block 和 TMEM 范围 ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
                        while tile_scheduler.valid():
                            ld2mma.wait(warp_id, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, warp_id * MMA_N:warp_id * MMA_N + MMA_N],
                                    Asmem[mma_ps.stage, warp_id, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                mma_ps.advance()

                            mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                            tile_scheduler.next_tile()

        # =============================================
        # Warpgroup 0/1: 写回（每个读取其消费者的 TMEM 范围）
        # =============================================
        elif wg_id < NUM_CONSUMER:
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((EPI_N,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wg_id, wb_ps.phase)  # 等待此消费者
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                # 以 EPI_N=64 列为块读取 TMEM（256 列需要 4 次迭代）
                for i in T.unroll(MMA_N // EPI_N):
                    reg = T.alloc_local((EPI_N,), acc_type)
                    reg_wg = reg.view(128, EPI_N,
                        layout=TileLayout(S[(128, EPI_N) : (1@tid_in_wg, 1)]))
                    col_st = T.meta_var(wg_id * MMA_N + i * EPI_N)
                    col_end = T.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
                    Tx.wg.copy_async(reg_wg[:], tmem[:, col_st:col_end])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if warp_id == 0:
                        if lane_id == 0:
                            m_st_epi = T.meta_var(
                                (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                            n_st_epi = T.meta_var(n_idx * MMA_N + i * EPI_N)
                            Tx.copy_async(
                                D[m_st_epi:m_st_epi+BLK_M, n_st_epi:n_st_epi+EPI_N],
                                Dsmem[wg_id, :, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(wg_id + 10)

                ld2mma.arrive(wg_id, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- 清理 ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**实现说明。**

- 在第 9 步的设计中，`mma2ld` 和 `ld2mma` 各自是单个共享对象，具有 `depth=NUM_CONSUMER`，而非每个消费者独立的对象。槽位 0 连接 MMA warp 0 到 Warpgroup 0，槽位 1 连接 MMA warp 1 到 Warpgroup 1；MMA 侧按 `warp_id` 索引，写回侧按 `wg_id` 索引。

## 端到端结果

下表报告了从朴素基线到 warp 特化 cluster kernel 的测量里程碑，以及 cuBLAS 参考结果。参考数据在 NVIDIA B200 上，M=N=K=4096，fp16，锁定时钟，1000 次迭代计时基准测试：

| 步骤 | 技术 | 时间 | 加速比 |
|------|-----------|------|---------|
| 1 | 同步加载 + MMA | 70 ms | 1× |
| 2 | K 循环累加 | --- | 处理 K 大于一个 tile 的情况 |
| 3 | 空间 tiling | 53.6 ms | ~1.3× |
| 4 | TMA 异步加载 | 0.49 ms | ~142× |
| 5 | 软件流水线 | --- | 重叠加载 + 计算 |
| 6 | 持久化 kernel | --- | L2 缓存局部性 |
| 7 | Warp specialization | 0.23 ms | ~309× |
| 8 | 2-CTA cluster | 0.104 ms | ~676× |
| 9 | 多消费者 | 0.094 ms | ~744× |
| --- | cuBLAS（参考） | 0.094 ms | ~744× |

此表中的每次计时，包括 70 ms 的第 1 步基线，都是在相同的 M=N=K=4096 规模下测量的，这使得加速比链条可以端到端地进行比较。值得精确说明 70 ms 实际是什么，因为它很容易被误读。它*不是*来自 {ref}`chap_gemm_basics` 的、在 4096³ 规模运行的单个 tile 的第 1 步 kernel；那个 kernel 只计算一个 128×128 tile，且只能在小规模下运行。70 ms 是一个朴素的完整规模基线，采用相同的串行、单 tile 方法并将其扩展到完整的 4096³ 问题。第 1-3 步在 {ref}`chap_gemm_basics` 中以小规模（128×128 和 256³）介绍，以保持这些初始步骤的简单性；此处的第 1 步和第 3 步行是它们在完整规模下的基准测试对应值。其余的破折号（第 2、5、6 步）标记了展示结构但未单独计时的步骤。

将这些数据视为受控条件下单次 B200 参考运行，而非排行榜条目。每个步骤中嵌入的 `{.python .input}` 基准测试单元是冒烟基准：适合发现趋势，不适合声称峰值性能。

四项技术几乎占据了所有收益：

1. **TMA 异步数据搬运**：硬件拷贝引擎替代软件拷贝（第 1 步 → 第 4 步约 142×）。正确解读这个 142× 很重要：它反映了从单个 128×128 tile kernel（grid 1×1）一直扩展到带有 K 循环、空间 tiling 和许多 CTA 的完整 tiling 和并行 kernel，*同时使用了 TMA*；它不是 TMA 单独贡献的隔离测量。隔离 TMA 意味着比较两个仅在拷贝机制上不同的完整规模 kernel。
2. **软件流水线 + Warp Specialization**：通过为加载和计算各自分配专用角色来重叠它们（第 4 步 → 第 7 步约 2.2×）。
3. **CTA Cluster**：2-SM 协作 MMA 改善跨 CTA 的 B-tile 重用（第 7 步 → 第 8 步约 2.2×）。
4. **多消费者**：两个 MMA warp 提高计算密度（第 8 步 → 第 9 步约 10%）。

以测量里程碑绘制，这四项贡献追踪了从同步 tiling kernel 到 cuBLAS 参考的下降过程。下图显示选定的测量点：

![GEMM 优化之旅](../img/gemm_perf.png)

注意收益随着我们沿列表向下递减，这有结构性原因而非努力减弱。早期步骤针对*内存*瓶颈（TMA 替代软件拷贝，cluster 提高算术强度），而 70 ms 中大部分时间实际上花在了这些地方，因此这些步骤收益最大。到第 8 步时，kernel 已经达到 cuBLAS 约 10% 以内（0.104 vs 0.094 ms），接近*计算绑定*，这意味着几乎没有剩余的内存停顿可以隐藏；第 9 步的多消费者重叠收回了剩余的少量收益。约 10% 的最终收益正是在接近计算上限时应该期待的：这是一个接近解决的剩余问题的边缘收益，而非弱优化的标志。

我们在本章构建的一切（TMA 加载、`tcgen05` MMA、TMEM 回读和 warp 特化 barrier）都直接延续到下一章。Flash Attention 复用了所有这些，然后通过在两个 MMA 阶段之间楔入在线 softmax 步骤而非简单重复单个 MMA 来提高难度。


## 练习

1. 如果在第 7 步中将 TMA 和 MMA 的 `PipelineState` 初始 `phase` 都设置为 `0`，会发生什么？画出死锁场景。
2. 在第 8 步 `cta_group=2` 时，TMA arrive 字节计数为 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`。为什么每个 CTA 加载自己的数据却要乘以 `CTA_GROUP`？
3. 在第 9 步中，每个消费者处理不同的 M 行但相同的 B tile。为什么共享 B（而不是 A）是正确的选择？

**与你的 agent 一起尝试**：粘贴第 7 步 kernel，让它追踪一个 K-tile 通过四个 barrier（`tma2mma`、`mma2tma`、`mma2ld`、`ld2mma`）的过程。对每个 barrier，问谁在等待，谁在到达，什么 tile 变得可以安全读取，之后哪个缓冲区变得可重用。
