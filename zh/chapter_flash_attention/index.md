(chap_flash_attention)=
# Flash Attention 4

:::{admonition} 概览
:class: overview

- Attention 运行两个 MMA，中间夹着 softmax，因此不能像 GEMM 那样简单重复一个 MMA。
- 该 kernel 组合了第一部分中的硬件原语（TMA、`tcgen05`、TMEM、barrier）和第三部分中 GEMM 技术的 warp 角色、在线 softmax 重缩放、因果 masking 和 GQA。
:::

Attention 是决定 transformer 能否运行的 kernel，也是我们迄今为止构建的一切必须协同工作的地方。我们为 GEMM 组装的每个部件都延续到了这里：TMA tile 搬运、`tcgen05` MMA、TMEM、warpgroup 寄存器 tile 和显式 barrier。

挑战在于 attention 不是一个 MMA 的重复。它是两个 MMA，中间夹着真正的工作：在线 softmax、因果 masking，以及保持前后 block 处于同一尺度的重缩放。

中间阶段是新的难点所在。普通矩阵乘法只需向其累加器添加；而 attention 必须在新的 key 和 value 流入时重新访问并重缩放已经计算的结果。softmax 工作本身也在两个 Tensor Core MMA 之间的 CUDA 核心上运行，因此指数运算和逐行规约直接处于关键路径上。

这就是为什么 attention 优化在很大程度上就是 softmax 优化：重新表述 `exp`，并将 softmax 与 MMA 重叠而不是在其上停顿。

本章的目标不是从头重新推导 Flash Attention。我们将只保留足够理解 kernel 所需的算法内容，然后将注意力花在真正新的部分：该算法如何变成 TIRx。

最清晰的切入方式是跟踪单个 tile 在 kernel 中的流动。`Q`、`K` 和 `V` 作为输入 tile 进入，从 GMEM 加载到 SMEM。分数 MMA 将 `Q` 和 `K` 相乘产生 TMEM 中的分数 tile `S`。Softmax 将 `S` 转换为分子 tile `P`，value MMA 将 `P` 和 `V` 组合以更新输出累加器 `O`。

到目前为止，这看起来像是两个矩阵乘法粘在一起，但有一个 GEMM 从未需要处理的转折：每当运行 softmax 最大值发生变化时，迄今为止累积的 `O` 突然处于错误的尺度中。它必须在下一个 value MMA 可以安全地添加到其中之前进行重缩放。下面各节首先追踪这条路径，然后展示 TIRx 如何将每个阶段交给一个 warpgroup 并将这些阶段连接起来。

## 算法形状

在我们能将 tile 放入内存之前，我们需要这些 tile 所服务的算法。对于一个 query block，Flash Attention 计算：

$$O = \text{softmax}(QK^{\top} / \sqrt{d})V$$

按字面理解，这个公式要求先形成完整的分数矩阵 `S = QKᵀ`，对其 softmax，然后乘以 `V`。这是我们唯一不能使用的方法，因为完整的 `S` 是巨大的。在 seq=4096 时，每个头大约存储 16M 个元素，在 fp32 下约 64 MB，比 SMEM 或单个 128×512 TMEM 区域大几个数量级。根本没有地方在芯片上存放它。Flash Attention 的答案是永远不具体化 `S`。相反，它以块的形式流式传输 `K/V`，并携带三个逐行运行状态来总结迄今看到的一切：

- `row_max`：迄今看到的最大分数。
- `row_sum`：softmax 的运行分母。
- `O`：运行输出累加器。

流式更新是使这些状态在新 block 到达时保持正确的关键。微妙之处在于，每次我们处理一个 block 时，运行最大值可能上升，一旦上升，我们在旧最大值下计算的所有内容现在都处于错误的尺度中。因此，在添加新贡献之前，我们首先将旧状态拉回到新的尺度中：

```text
S = Q_block @ K_block.T
m_new = max(row_max, rowmax(S))
scale = exp((row_max - m_new) / sqrt(d))
P = exp((S - m_new) / sqrt(d))
row_sum = row_sum * scale + rowsum(P)
O = O * scale + P @ V_block
row_max = m_new
```

单个 `scale` 因子在这里起到双重作用：它同时重缩放运行分母和运行输出，使得来自更早和更晚 block 的贡献最终以统一的尺度度量。

上面的伪代码是用自然的 `exp` 和显式的 `/sqrt(d)` 编写的，因为这样最易读，但 kernel 采用了更便宜的方式。它将 `1/sqrt(d)` 和 `log2(e)` 折叠成一个常数 `scale_log2 = log2(e)/sqrt(d)`，并使用硬件 `exp2` 在原始分数上评估每个指数，利用恒等式 `exp(x/sqrt(d)) = exp2(x · scale_log2)`。动机很简单，在此硬件上 `exp2` 比自然的 `exp` 更快。

在继续之前有一点值得明确：这里的 `P` *不是*最终归一化的 attention 矩阵。它只是当前 K/V block 的 softmax 分子。归一化被有意延迟了，只有在最后一个 block 之后 kernel 才写入 `O / row_sum`。

对于 TIRx，知道算法计算什么只是图片的一半。另一半是*每个 tile 在 kernel 运行时的驻留位置*，因为这是决定布局和 barrier 代码的关键。`S`、`P` 和 `O` 都是 tile 值，每个都有一个归属：

- `S` 是分数 tile。分数 MMA 将其写入 TMEM。
- `P` 是 softmax 分子 tile。Softmax 从 TMEM 读取 `S` 到寄存器，计算 `P = exp((S - m_new) / sqrt(d))`，并将 `P` 写回 TMEM。
- `O` 是输出累加器 tile。Value MMA 从 TMEM 读取 `P`，从 SMEM 读取 `V`，然后累加到 TMEM 中的 `O`。

我们之前标记的重缩放也是一个 tile 操作，而非标量簿记：当 `row_max` 改变时，旧的 `O` 从 TMEM 中读取，在寄存器中乘以系数，并写回 TMEM，然后下一个 value MMA 才累加到其中。后续每一节都遵循相同的结构：tile 放置、硬件路径和证明下一个消费者可以运行的 barrier。

## Tile 原语图

有了运行状态及其归属，我们可以将算法布局为 tile 移动的具体序列。对于一个 K/V block，kernel 从上到下遍历此 tile 路径：

```text
Q, K, V in GMEM
  -> Q, K, V in SMEM        通过 TMA 加载
  -> S in TMEM              通过分数 MMA：QK^T
  -> P in TMEM              通过 softmax 分子：TMEM -> RF -> TMEM
  -> O in TMEM              通过 value MMA：P V
  -> O in GMEM              通过归一化、SMEM 暂存和 TMA 存储
```

与 GEMM 的区别归结为一行。GEMM 是一个 MMA 链的重复；FA4 有两个 MMA 阶段，softmax 位于链中间。几乎所有其他内容都是这一额外阶段的结果。

如果我们将短路径扩展为显式的生产者-消费者边，就得到了完整的图：

| 阶段 | Tile 移动或计算 | TIRx 原语 | 硬件路径 |
|-------|--------------------------|----------------|---------------|
| 加载 Q/K/V | GMEM tile -> SMEM tile | `Tx.copy_async(..., dispatch="tma")` | TMA 加载 |
| 分数 MMA | SMEM 中的 Q 和 K -> TMEM 中的分数 tile `S` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` |
| Softmax 读取 | TMEM 中的 `S` -> warpgroup 寄存器 tile | `Tx.wg.copy_async(reg, tmem)` | `tcgen05.ld` |
| Softmax 写入 | 寄存器中的分子 tile `P` -> fp16 TMEM 视图 | `Tx.copy_async(tmem_as_f16, reg)` | TMEM 存储，后跟 `tcgen05.wait.st()` |
| Value MMA | TMEM 中的 `P` 和 SMEM 中的 V -> TMEM 中的输出累加器 `O` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | 带有 TMEM 操作数的 `tcgen05.mma` |
| 修正 | TMEM 中的 `O` -> 寄存器 -> TMEM 中的 `O` | TMEM 回读、寄存器乘法、TMEM 存储 | `tcgen05.ld` / TMEM 存储 |
| Epilogue | TMEM 中的最终 `O` -> 寄存器 -> SMEM -> GMEM | TMEM 回读、`Tx.copy`、TMA 存储 | `tcgen05.ld` + TMA 存储 |

新行是 softmax 和修正。两者都增加了 TMEM → 寄存器 → TMEM 的流量，两者都在分数 MMA 和 value MMA 之间创建了额外的握手。

**与你的 agent 一起尝试**：让它只追踪上面的短路径。对每个箭头，命名生产者阶段、消费者阶段、源 tile、目标 tile 和硬件路径。然后问哪些箭头在 GEMM 章节中不存在。

## Warp 角色和范围

数据路径确定后，下一个自然的问题是实际上谁运行每个阶段。这里的每个 CTA 有 4 个 warpgroup，共 512 个线程，它们不是按它们触摸的数据来划分，而是按 warpgroup 执行的*工作类型*来划分：

- WG3 驱动硬件引擎：TMA 加载、MMA 和 TMA 存储。
- WG0、WG1 和 WG2 执行这些引擎调用之间的寄存器密集型数学运算：softmax、修正和 epilogue。

确切的角色表是：

| 所有者 | 角色 | 做什么 |
|-------|------|--------------|
| WG3, warp 1 | TMA 加载 | 将 Q、K 和 V tile 从 GMEM 加载到 SMEM |
| WG3, warp 0 | MMA | 发出分数 MMA 和 value MMA |
| WG3, warp 2 | TMA 存储 | 将最终 O tile 从 SMEM 存储到 GMEM |
| WG0 | Q 阶段 0 的 Softmax | 从 TMEM 读取 S，计算 P，将 P 写入 TMEM |
| WG1 | Q 阶段 1 的 Softmax | 对第二个 Q 流水线阶段做相同工作 |
| WG2 | 修正和 epilogue | 重缩放 TMEM 中的 O，归一化，暂存输出 |

很容易将"两个 Q 阶段"误读为两个 attention 头，但它们不是。它们只是 Q 流水线中的两个槽位，WG0 拥有一个，WG1 拥有另一个，以便两个 Q tile 可以同时处理中。这就是 softmax 工作出现两次的原因，一次在 WG0 上，一次在 WG1 上。

代码使用符号坐标选取这些角色：

```python
wg_id = T.warpgroup_id([4])
warp_id = T.warp_id_in_wg([4])
```

当你阅读 kernel 时，首先找到角色分支。它告诉你哪个团队拥有嵌套在其中的每个 tile 原语。

- WG3 warp 1 启动 TMA 加载命令。一个选中的 lane 发起拷贝，TMA 引擎搬运 tile。
- WG3 warp 0 发出 `tcgen05.mma` 指令。
- WG0 和 WG1 在全 warpgroup 范围内运行 softmax。
- WG2 在全 warpgroup 范围内运行修正和 epilogue 工作。

一个不对称性最终塑造了整个 barrier 图：*每个* MMA，无论是分数 MMA 还是 value MMA，都仅从 WG3 warp 0 发出。WG0 和 WG1 根本不发出 MMA。它们只消费分数 tile、运行 softmax 并将 `P` 写回 TMEM。

这种分离正是 softmax 需要围绕它设置 barrier 的原因。`s_ready` 将分数 tile 从 MMA warp 传递到 softmax；`p_o_rescale` 传递 `P` 和一个对 value MMA 安全的 `O` 槽位（要么已经重缩放，要么因不需要重缩放而释放）。我们将在本章剩余部分不断回到这两个名称。

## 阅读代码片段

本章的代码片段摘录自 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py)，因此它们不可避免地引用了 kernel 中我们没有复现的部分所定义的名称。自描述的名称（`wg_id`、`warp_id`、`BLK_M`/`BLK_N`、`HEAD_DIM`、`kv_stage`、`SMEM_PIPE_DEPTH_*` / `TMEM_PIPE_DEPTH` 深度、`should_accumulate`，以及 `CTA_GROUP`（此处为 1））我们在它们首次出现时介绍。其余的在下表中有一行注解，这样当一个片段将一个不熟悉的名称放在你面前时，你就有了可以查阅的地方：

| 名称 | 含义 |
|------|---------|
| `q_stage`、`i_q` | Q 流水线阶段，0 或 1，即哪个 Q tile 槽位（`SMEM_PIPE_DEPTH_Q = 2`）。在 WG0/WG1 softmax 内部，warpgroup 自己的 `wg_id`（0 或 1）*就是*这个相同的阶段索引，因此 `S_region[q_stage]`、`P_region[wg_id]` 和 `O_region[i_q]` 都选择相同的 Q 阶段 |
| `MMA_N` | TMEM 列中分数/输出 tile 宽度（128） |
| `MMA_K` | `P`/`V` 列中的 MMA 内 K 步（16）；`K_SPLIT = 6 * MMA_K = 96` |
| `K_SPLIT` | value-MMA 调度的分割点（参见*两个 MMA 阶段*）；第一个 value MMA 覆盖列 `0:K_SPLIT`（`6 * MMA_K = 96`） |
| `should_rescale` | WG2 逐行标志：在下一个 value MMA 之前旧 `O` 是否需要重缩放（通过 `any_sync` 跨 warpgroup 规约） |
| `rescale_threshold` | 小行最大变化的跳过阈值；当前 kernel 使用 `8.0`，跳过的重缩放将 `acc_scale` 设置为精确的 `1.0` |
| `scale_log2` | 以 log2 为单位的 softmax 缩放，`log2(e)/√d`，因此 `P = exp2((S - m) · scale_log2)` |
| `acc_scale` | softmax 通过 SMEM 邮箱传递给 WG2 的逐行重缩放因子 |
| `chunk_start`/`chunk_end`、`p_start`/`p_end` | 正在读/写的 32 宽 softmax 块的列范围 |

## 两个 MMA 阶段

对于每个流式传输的 K/V tile，Flash Attention 运行两个 MMA 阶段，softmax 桥接它们：

```text
Q, K -> 分数 MMA -> S
S    -> softmax   -> P
P, V -> value MMA -> O
```

将其视为一行中三个生产者的流水线。第一个 MMA 产生 attention 分数 `S`，softmax 将 `S` 转换为分子 `P`，第二个 MMA 消费 `P` 来更新输出累加器 `O`。通过 `row_sum` 的归一化被推迟到 epilogue，即每个 K/V tile 都发挥了作用之后。

下面的每个 tile 操作都获得我们在 GEMM 步骤中使用的相同的**范围 / 布局 / 分发**卡片，并增加了一行额外的**握手**，命名将 tile 传递给下一个角色的 barrier(s)。

计算代码从不使用原始的 TMEM 列号。相反，kernel 将其单一的 TMEM 分配划分为每个阶段的视图（`S_region`、`P_region`、`O_region`），并按流水线阶段索引它们（`S_region[q_stage]`、`O_region[i_q]`、`P_region[i_q, 0:K_SPLIT]`）。这些视图在 [TMEM 布局和复用](#tmem-layout-and-reuse) 部分中用 `T.TMEMStages` 定义；目前只需将每个区域视为同一物理 TMEM 的一个命名切片即可。

### 分数 MMA

两个阶段中的第一个是分数 MMA，即开启每次 K/V 迭代的矩阵乘法。它计算：

$$S = Q_{\text{block}}K_{\text{block}}^{\top}$$

并将 `128 x 128` 的分数 tile 写入 TMEM：

```python
Tx.warp.gemm_async(
    S_region[q_stage],
    Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
    K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
if T.ptx.elect_sync():
    s_ready.arrive(q_stage)
```

我们可以提出 GEMM 章节对每个 tile 操作提出的相同的四个问题：谁运行它，tile 在哪里，它如何分发，以及它如何握手：

> **Tile 原语解读：分数 MMA**
> - 范围：WG3 warp 0 发出它；一个选中的 lane 到达 `s_ready`。
> - 布局：Q、K 在 SMEM 中 → `S` 在 TMEM 中（`S_region[q_stage]`）。
> - 分发：`tcgen05`。
> - 握手：`s_ready`（→ softmax）。

单个选中线程到达 `s_ready` 就是整个握手。它宣布此分数 tile 已完成，softmax warpgroup 现在可以自由地读取它。

### 两个 MMA 之间的 Softmax

在两个 MMA 之间的是 softmax，即将分数 tile `S` 转变为分子 tile `P` 的阶段。它的解读卡是：

> **Tile 原语解读：Softmax**
> - 范围：WG0（Q 阶段 0）/ WG1（Q 阶段 1），全 warpgroup。
> - 布局：TMEM 中的 `S` → 寄存器 → fp16 TMEM 中的 `P`（`P_region[wg_id]`）。
> - 分发：`tcgen05.ld` 读取，TMEM 存储写入；两者之间的逐行寄存器数学运算。
> - 握手：等待 `s_ready`；到达 `p_o_rescale`（前 96 列）和 `p_ready_2`（最后 32 列）。

此阶段在 GEMM 中完全没有对应物。WG0/WG1 等待分数 tile 在 `s_ready` 上到达，然后一次一个寄存器大小的块从 TMEM 读取它：

```python
Tx.copy_async(
    s_chunk[:, chunk_start : chunk_end],
    S_region[wg_id, chunk_start : chunk_end],
)
```

这是在 warpgroup 范围下的 TMEM 到寄存器 tile 读取。现在分数已驻留在寄存器中，softmax warpgroup 按顺序做三件事：

1. 计算行最大值和行和，
2. 计算 softmax 分子 tile `P`，
3. 将 `P` 作为 fp16 写回 TMEM。

最后一步看起来像：

```python
Tx.copy_async(
    P_region[wg_id, p_start : p_end],
    p_chunk[:, p_start : p_end],
)
```

既然我们刚在寄存器中完成了计算，为什么还要将 `P` 写回 TMEM？因为 value MMA 需要 `P` 作为 *tile 操作数*，而 MMA 不能将分散的每线程标量寄存器作为矩阵读取。此 kernel 中 MMA 可读形式的 `P` 是 `P_region`，即 fp16 TMEM 别名 `tmem_as_f16` 上的视图。因此，写回并不是多余的移动；它是将 `P` 放入下一个 MMA 实际可以消费的唯一形状的关键步骤。

### Value MMA

第二个阶段，也是结束每次 K/V 迭代的阶段，是 value MMA。它计算：

$$O = O + P_{\text{block}}V_{\text{block}}$$

到此 MMA 运行时，`O` 已经为当前 K/V block 置于正确的状态（在第一个 block 时初始化，在后续 block 时重缩放），因此 MMA 只需累加。与 GEMM 的不同之处在于操作数的位置：A 操作数是 TMEM 中的 `P`，B 操作数是 SMEM 中的 `V`，累加器 `O` 也在 TMEM 中：

```python
# 第一个子 MMA：列 0:K_SPLIT（P 的前 96 列 / V 的行）。
Tx.warp.gemm_async(
    O_region[i_q],
    P_region[i_q, 0:K_SPLIT],
    V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
    transB=True,
    accum=should_accumulate,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
# 第二个子 MMA（相同形式，accum=True，由 p_ready_2 门控）覆盖
# 剩余的列 K_SPLIT:BLK_N。
```

> **Tile 原语解读：Value MMA**
> - 范围：WG3 warp 0。
> - 布局：TMEM 中的 `P` + SMEM 中的 V → TMEM 中的 `O`（`O_region[i_q]`）。
> - 分发：`tcgen05`，带有 TMEM 操作数。
> - 握手：等待 `p_o_rescale`、`p_ready_2`、`kv_load.full`；到达 `o_ready`（→ epilogue）。

这个操作数放置是两个 MMA 之间的硬件差异：

- 分数 MMA 从 SMEM 读取两个操作数：Q 和 K。
- Value MMA 从 TMEM 读取一个操作数 `P`。
- Value MMA 从 SMEM 读取另一个操作数 V。
- 结果累加到 TMEM 中的 `O`。

`accum=should_accumulate` 标志实现了算法中的"初始化或添加"选择：它对 query block 的第一个 K/V tile 为 false，对之后的每个 tile 为 true。

你可能还注意到 value MMA 不是一次运行，而是拆分为 `96 + 32` 的调度：

1. Softmax 分四个 32 列块写入 `P`。
2. 一旦前三个块就绪，value MMA 就开始处理 `P` 的前 96 列和 V 的匹配行。
3. 最后 32 列等待 `p_ready_2`。
4. 第二个 MMA 消费最后一个块并完成 tile。

拆分的原因是为了保持 Tensor Core 忙碌。如果将 value MMA 作为单条指令运行，整个阶段将停滞，直到所有四个 32 列 `P` 块都完成了指数运算和存储。通过立即开始前三个块，kernel 将最后一个块的 `exp` 和 TMEM 写入与已经在运行中的 96 宽 MMA 重叠，将原本闲置的时间转变为有用的工作。

## TMEM 布局和复用

所有 `S`、`P` 和 `O` 必须共享一个 `128 x 512` 的 TMEM 分配，它们被打包进去的方式正是 barrier 和布局在此 kernel 中不可分割的原因：

下图直接展示了这种打包：分数槽位、分子槽位和输出槽位都
共享一个 TMEM 分配，因此 barrier 协议是使复用合法的关键。

![TMEM 布局](../img/tmem_layout_v3.png)

该图读作一组 tile 槽位：

- 分数槽位保存 `S = QK^T`。
- 分子槽位保存 softmax 指数运算步骤后的 `P` tile。
- 输出槽位保存 fp32 的 `O` 累加器。

这些不是独立的缓冲区。它们是*同一*分配的区域，共享不是风格选择而是被迫的。在 Q 流水线深度为 2 的情况下，两个 `S` 槽位（2 × MMA_N = 256 列）和两个 `O` 槽位（2 × MMA_N = 256 列）已经占了所有 512 个 fp32 列。没有余量留给 `P`，所以 `P` 别无选择，只能通过一个较窄的 fp16 视图来别名化相同的字节。这之所以安全，唯一的原因是因为每个区域严格在其前一个消费者完成后才被复用，而这个时序正是 barrier 所保证的。因此在 FA4 中，barrier 不仅仅是调度；它们首先是使布局合法化的关键。

别名化技巧通过 `T.TMEMPool` 设置。kernel 获取一个 fp32 视图（`tmem`）用于分数和输出累加器，然后将池基地址回退到 0，获取第二个 fp16 视图（`tmem_as_f16`）覆盖*相同的*物理字节：

```python
tmem_pool = T.TMEMPool(pool, total_cols=N_COLS_TMEM, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_pool.move_base_to(0)
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()
```

因为 fp16 元素宽度减半，fp16 视图在相同字节上暴露两倍的可索引列，而这正是 `P` 所在的空间，是 fp32 布局没有空间容纳的空间。有了两个视图，kernel 用 `T.TMEMStages` 将 `S`、`P` 和 `O` 槽位划分成分阶段的区域，这使得计算代码可以按流水线阶段索引而非原始列号：

```python
S_region = T.TMEMStages(tmem,        col_start=0,                       width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
O_region = T.TMEMStages(tmem,        col_start=MMA_N * SMEM_PIPE_DEPTH_Q, width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
P_region = T.TMEMStages(tmem_as_f16, col_start=MMA_N,                   width=BLK_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N * 2)
```

`P_region` 步幅中的 `* 2` 是别名化在代码中可见泄漏的唯一地方。`S_region` 和 `O_region` 以 fp32 的 `tmem` 列度量，而 `P_region` 以 fp16 的 `tmem_as_f16` 列度量，后者宽度减半，因此阶段间的移动需要双倍的步幅才能落在相同的物理字节上。然而一旦区域定义好之后，计算代码保持清爽：它写入 `S_region[q_stage]`，读取 `S_region[wg_id, ...]`，写入 `P_region[wg_id, ...]`，累加到 `O_region[i_q]`，从不触碰原始列索引。

**与你的 agent 一起尝试**：让它解释此 FA4 kernel 中的 fp32（`tmem`）和 fp16（`tmem_as_f16`）视图。哪些物理 TMEM 区域保存 `S`、`P` 和 `O`，为什么 `P_region` 的步幅使用 `MMA_N * 2`？将复用问题留到下一节：在 barrier 表之后，检查每个区域可以复用之前哪些消费者必须完成。

## Barrier 如何连接角色

这是 kernel 中最难的部分，所以值得逐步深入。从沿主计算路径移动数据的少数几个 barrier 开始，将其他一切都视为你可以稍后查阅的簿记。数据就绪握手是：

| 握手 | 含义 |
|---------|---------|
| TMA 加载 -> 分数/value MMA | Q、K 或 V 已到达 SMEM 并可以输入 MMA |
| 分数 MMA -> softmax | TMEM 中的 `S` 已就绪 |
| softmax/修正 -> value MMA | TMEM 中的 `P` 已就绪，且 `O` 对累加安全 |
| value MMA -> epilogue | TMEM 中的最终 `O` 已就绪 |
| epilogue -> TMA 存储 | `O_smem` 已准备好存储 |

不在该列表中的所有内容都是流水线簿记：释放 SMEM、TMEM 或暂存缓冲区以便另一个角色可以复用它们的 barrier。有用的是，每个 barrier，无论它携带数据还是仅做簿记，都以相同的方式读取，作为一个 tile 握手。你问谁生产了数据，谁消费了它，以及一旦两者都完成后哪个缓冲区变得空闲。

下图将这些握手折叠为两个 MMA 阶段的确切就绪门控：
分数 MMA 等待什么，以及 value MMA 在可以累加之前必须等待什么。

![Flash Attention 4 MMA 输入门控](../img/flash_attention_main_handoff.png)

将此图读作一组正确性门控而非调度。它回答了"在该 MMA 可以发射之前什么必须为真"，而不涉及时序。分数 MMA 等待 SMEM 中的 Q 和 K，然后产生 `S`。Value MMA 同时等待三件事：SMEM 中的 V、来自 softmax 的 `P` tile，以及 WG2 已经释放或重缩放的 `O` 槽位。softmax 到 value 的门控被拆分，原因我们已经见到了：value MMA 可以在 `P` 的前 96 列就位后立即开始，`p_ready_2` 释放最后 32 列。

有一个握手不符合 tile 就绪的模型：softmax 到修正的边。不是传递 tile，softmax 通过一个槽位的 SMEM 邮箱向 WG2 传递单个标量（K/V 循环中的 `acc_scale`，或 epilogue 中的最终 `row_sum`）。由于该槽位在每次迭代时被复用，需要一对 `full`/`empty` barrier 来守卫它：

下图放大显示了那个邮箱握手机制，这就是为什么这一对 barrier 应该
被读作标量生产者-消费者通道而非 tile 就绪门控。

![Flash Attention 4 Softmax 尺度槽位握手](../img/flash_attention_softmax_correction.png)

将 `softmax_corr.full` 和 `softmax_corr.empty` 读作一对生产者-消费者：

1. Softmax 在复用尺度/和槽位之前等待 `softmax_corr.empty`。
2. Softmax 将 `acc_scale` 或最终 `row_sum` 写入该槽位。
3. Softmax 到达 `softmax_corr.full`。
4. WG2 等待 `softmax_corr.full`，然后读取该槽位。
5. WG2 到达 `softmax_corr.empty`。
6. Softmax warpgroup 可以在下一阶段复用该槽位。

值得仔细区分 `softmax_corr.empty` 的含义和不含义。它只表示 WG2 已消费了尺度/和槽位。它完全不涉及 `P` 是否就绪，并且它绝对*不是*让 value MMA 启动的门控。那个门控是 `p_o_rescale`，它在 `P` 的前 96 列写入完成且 `O` 槽位对累加安全时触发。混淆两者是错误结果 bug 的经典来源。

有了主路径在手，完整的 barrier 列表作为参考：

| Barrier | 生产者 -> 消费者 | 什么变得安全 |
|---------|----------------------|-------------------|
| `q_load.full` | TMA 加载 -> 分数 MMA | Q SMEM tile 可以输入 MMA |
| `q_load.empty` | 此 Q 阶段的所有分数 MMA -> TMA 加载 | Q SMEM 阶段可以重用于下一个任务 |
| `kv_load.full` | TMA 加载 -> 分数/value MMA | K 或 V SMEM tile 可以输入 MMA |
| `kv_load.empty` | 分数/value MMA -> TMA 加载 | K/V SMEM 阶段可以复用 |
| `s_ready` | 分数 MMA -> softmax | S TMEM tile 可被读取 |
| `p_o_rescale` | softmax + WG2 -> value MMA | P 的前 96 列在 TMEM 中，O 槽位对 value MMA 安全 |
| `p_ready_2` | softmax -> value MMA | P 的最后四分之一在 TMEM 中 |
| `o_ready` | value MMA -> epilogue | 最终 O 累加器已就绪 |
| `softmax_corr.full` | softmax -> WG2 | `acc_scale` 或最终 `row_sum` 在 SMEM 邮箱中就绪 |
| `softmax_corr.empty` | WG2 -> softmax | 相同的 SMEM 邮箱槽位在 WG2 读取后可复用 |
| `corr_epi.full` | epilogue -> TMA 存储 | O_smem 已准备好存储 |
| `corr_epi.empty` | TMA 存储 -> epilogue | O_smem 阶段可复用 |

正如在 GEMM 中一样，你可以通过谁产生信号来预测 barrier 的类型：

- TMA 加载使用 `TMABar`，因为 TMA 引擎自己计算字节完成。
- MMA 完成使用 `TCGen05Bar`，因为 `tcgen05.commit` 通知完成组。
- 纯线程到线程握手使用 `MBarrier`，参与线程显式到达。

拆分后的 softmax 到 value 握手值得更仔细地观察。它使用两个门控：

- `p_o_rescale` 让 value MMA 在 `P` 的前 96 列写入完成且 `O` tile 对累加安全后开始。
- `p_ready_2` 释放 `P` 的最后 32 列，匹配前一节的 `96 + 32` value-MMA 调度。

第一个 K/V block 是简单情况。WG2 预先到达 `p_o_rescale`，因为还没有旧的 `O` tile 需要重缩放。

后续 block 必须更小心。WG2 只有在其跳过不必要的重缩放或完成重缩放旧 `O` 之后才到达 `p_o_rescale`。跳过测试是故意保守的：softmax 计算 log2 缩放的增量 `(m_old - m_new) * scale_log2`；如果该值仍然高于 `-rescale_threshold`，新的最大值没有移动足够远来证明重缩放是必要的，所以 kernel 保留旧的最大值并将 `acc_scale` 设置为精确的 1.0。只有更大的最大值跳跃才走 `exp2` 路径并要求 WG2 重缩放 `O`。

然后 WG2 通过 `any_sync` 跨 warpgroup 规约 `should_rescale`。如果没有行需要更新，它就让 `O` 保持不变。该跳过很重要，因为重缩放 `O` 是对整个累加器的完整 TMEM → RF → TMEM 读-改-写，当阈值逻辑已经将 `acc_scale` 保持为 1.0 时，这是纯粹浪费的工作。

注意所有新的 barrier 都聚集在一个地方。`s_ready`、`p_o_rescale`、`p_ready_2` 和 softmax/修正 pair 都是 softmax 周围的 barrier。它们存在的原因只有一个：分数 MMA 和 value MMA 不再相邻。寄存器数学、TMEM 重写和输出重缩放现在位于它们之间，而这些步骤中的每一个都需要自己的握手。

**与你的 agent 一起尝试**：让它追踪一个 K/V block 通过 `s_ready`、`p_o_rescale`、`p_ready_2` 和 `o_ready`。对每个 barrier，问谁在等待，谁在到达，什么 tile 变得可以安全读取，之后什么存储可以复用。

## 流水线结构

barrier 告诉我们一个角色消费 tile 之前什么必须*就绪*。它们没有告诉我们实际什么在*并发*运行，而这是我们转向的问题。这两者确实不同：正确性门控可以在生产者碰巧运行之前或之后很久被满足。

这里没有单一的流水线深度，因为不同的 tile 流以不同的速率移动。因此 kernel 为每个保持单独的环形缓冲区：

- Q 流水线深度 2：一个 CTA 处理两个 Q 阶段。WG0 处理一个阶段，WG1 处理另一个。
- KV 流水线深度 3：K 和 V block 流式通过内部循环，而相同的 Q 阶段被复用。
- TMEM 流水线深度 2：每个 Q 阶段有自己的 S/P/O TMEM 槽位，这些槽位在匹配的 barrier 触发后被复用。

下图从正确性门控切换到时间线视图，展示了在这些单独的环形缓冲区启动后，哪些角色可以大致同时活跃。

![Flash Attention 4 流水线结构](../img/flash_attention_pipeline_v2.png)

将此读作时间线而非 barrier 图。它展示了哪些角色在大致相同的时刻活跃，而之前的 barrier 流图是你去检查确切的生产者-消费者等待的地方。两者之间，这两张图回答了我们本节开头提出的两个不同问题。

每一行匹配代码中的一个角色分支：

- WG3 warp 1 发出 TMA 加载。
- WG3 warp 0 发出分数 MMA 和 value MMA。
- WG0 和 WG1 为两个 Q 阶段运行 softmax。
- WG2 释放或重缩放 `O`，然后稍后归一化最终输出。
- WG3 warp 2 发出 TMA 存储。

从左到右跟随该图追踪一个代表性的流水线波。加载 warp 以 `Q0`、`K[n-1]`、`Q1`、`V[n-1]` 开始，然后继续流式传输更低索引的 K/V block。MMA warp 发出第一批分数 MMA 以产生 `S0` 和 `S1`，WG0/WG1 将其转换为 `P0` 和 `P1`。

重要的是，MMA warp *不*先运行所有分数 MMA 然后再运行所有 value MMA。一旦两个 Q 阶段被启动，它就交替进行两种类型：当前 `V` block 的 value MMA，然后是下一个 `K` block 的分数 MMA，以此类推：

```text
score Q0*K[n-1]
score Q1*K[n-1]
value P0*V[n-1]
score Q0*K[n-2]
value P1*V[n-1]
score Q1*K[n-2]
value P0*V[n-2]
...
```

这种交替是为什么图中分数、softmax、修正和 value 行全部重叠而不是按整齐的顺序运行。

WG2 行标为 `release / rescale`，两半对应我们看到的两种情况。在第一个 K/V block 上还没有旧的 `O`，因此 WG2 只参与让 value MMA 继续进行的握手；在后续 block 上，它可能在 value MMA 累加之前重缩放旧的 `O`。归一化和 TMA 存储恰好发生一次，在 attention 任务的最终 K/V block 之后。

没有单一的 GEMM 式流水线可以描述 FA4，因为 Q、K/V 和 TMEM 槽位都按独立的调度推进。TIRx 将这些调度保持为显式的，作为单独的 tile 缓冲区、`PipelineState` 游标和 barrier 相位，而不是将 kernel 隐藏在单个庞大原语后面。代价是更多的移动部件，但好处是复杂性保持可见和可检查。

## 重缩放和写回

重缩放是强制性的，不是我们可以放弃的优化。在线 softmax 可以随着每个新分数 tile 提高逐行最大值，而当它这样做时，来自更早 block 累积的 `O` 是用*旧*最大值缩放的。这使得每个更早的项都偏大了一个因子 `exp(m_new - m_old)`。跳过修正，那些 block 被过加权，最终输出就是错误的。修复是一个 TMEM → 寄存器 → TMEM 的 tile 操作：

$$O_{\text{old}} \leftarrow O_{\text{old}} \cdot e^{(m_{\text{old}} - m_{\text{new}}) / \sqrt{d}}$$

工作跨两个角色拆分。Softmax 计算逐行尺度并将其放入 SMEM 邮箱；WG2 等待 `softmax_corr.full`，从 TMEM 读取当前的 `O`，乘以该尺度，然后将 `O` 写回：

```python
RESCALE_TILE = T.meta_var(16)
o_row = T.wg_reg_tile(RESCALE_TILE)
Tx.copy_async(o_row, O_region[i_q, d_start : d_start + RESCALE_TILE])
Tx.mul(o_row, o_row, acc_scale)
Tx.copy_async(O_region[i_q, d_start : d_start + RESCALE_TILE], o_row)
T.ptx.tcgen05.wait.st()
```

值得强调的是，这是对整个 `O` 累加器的完整 TMEM → 寄存器 → TMEM tile 操作，而不是一些标量簿记，它携带与其他每个阶段相同的解读卡：

> **Tile 原语解读：修正（重缩放）**
> - 范围：WG2，全 warpgroup。
> - 布局：TMEM 中的 `O` → 寄存器 → TMEM 中的 `O`（`O_region[i_q]`）。
> - 分发：`tcgen05.ld` 读取，TMEM 存储写入；两者之间的寄存器乘法。
> - 握手：等待 `softmax_corr.full`；到达 `p_o_rescale`（→ value MMA）和 `softmax_corr.empty`（→ softmax）。

端到端追踪同步：

1. Softmax 将尺度值写入 SMEM。
2. WG2 等待 `softmax_corr.full`。
3. WG2 在 TMEM 中重缩放 `O`。
4. WG2 到达 `p_o_rescale`。
5. WG3 的 value MMA 现在可以消费 `P` 并累加到重缩放后的 `O` tile。

当 `softmax_corr.empty` 在 WG2 读取后释放 SMEM 槽位时，循环闭合，这释放了 softmax 在下次迭代时复用邮箱。

一旦 K/V 循环结束，WG2 从修正切换到 epilogue。它等待最终的 `row_sum` 和 `o_ready`，从 TMEM 读取最终的 `O`，乘以 `1 / row_sum`（我们一开始推迟的归一化），转换为 fp16，并写入 `O_smem`。然后 WG3 的 TMA 存储 warp 将 `O_smem` 带回 GMEM。

一个限制值得为计划扩展此 kernel 的人标记出来。它仅计算前向输出，而训练前向传递通常还会存储后向传递所需的 log-sum-exp（LSE）。添加它伴随着一个需要记住的缩放细节：此 kernel 将 `row_max` 保持为*原始的、未缩放的* `QK^T` 分数的最大值，而 `row_sum` 累积 `exp((S - row_max) / sqrt(d))`。所以在形成自然对数 LSE 时，`1/\sqrt{d}` 因子必须重新应用于 `row_max`：

$$\mathrm{LSE}_i = \log(\mathrm{row\_sum}_i) + \mathrm{row\_max}_i / \sqrt{d}$$

此实现仅提供前向输出，不写入 LSE。

## 因果 Masking

因果 attention 添加了一个约束（一个 query 只能 attention 到其自身位置或之前位置的 key），kernel 以两种互补的方式来满足，一种便宜，一种精确。

便宜的方式是完全跳过工作。许多 K/V block 完全位于对角线以上，对给定的 Q block 没有任何贡献，所以 `get_n_block_max(...)` 计算该 block 可能需要的最后一个 block，循环直接跳过其余部分而不加载或计算。

精确的方式处理跨越对角线的 block，其中一些列有效，另一些无效。这些 block 仍然运行分数 MMA，但 softmax 在指数运算之前屏蔽无效列。对于每一行，它从该行的 query 位置和 block 偏移导出一个列限制，保留该限制及以下的列，并将超出限制的每一列设置为寄存器中的 `-inf`，这样这些列对行最大值和 `exp2` 分子都没有贡献。

实现不是逐个元素分支，而是用 `mask_r2p(...)` 应用限制，它将限制转换为覆盖整个 32 宽分数块的位掩码，并一次性屏蔽该块。完全位于对角线以下的 block 保留每一列，完全不需要掩码。

从 tile 原语的角度看，因果模式根本不会重写数据路径。它只削减了 K/V 循环计数，并在驻留寄存器的 softmax 中（在分数 MMA 和 `P` 写回之间）插入了一个屏蔽步骤。

## GQA 支持

分组 Query Attention 允许多个 query 头共享一个 K/V 头。这节省了内存带宽，但提出了一个打包问题：我们如何只保留一个 K/V tile 同时仍然通过它喂送多个 query 头？kernel 的答案是一次处理针对一个调度的 `kv_head_idx` 的整组 query 头：

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

技巧在于重新解释 128 个 Q-tile 行。对于 `GQA_RATIO=4`，它们不再代表 128 个序列位置；它们代表 32 个序列位置乘以 4 个 query 头，打包在一起以便所有四个头使用同一个 K/V tile。行解码是：

```text
seq_pos = row // GQA_RATIO
q_head  = row % GQA_RATIO
```

Q 加载用 3D 视图表达这种打包。源是自然的 `Q[batch, seq, qo_head, dim]` 布局，而目标是分数 MMA 稍后将作为平坦 `128 x HEAD_DIM` 操作数读取的同一个 SMEM tile。视图是调和两者的关键，它在没有任何拷贝的情况下完成：

```python
Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
Tx.copy_async(
    Q_smem_3d[i_q, :, :, :],
    Q[batch_idx,
      m_start : m_start + SEQ_Q_PER_TILE,
      kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
      :],
    **tma_copy_q,
)
```

K 和 V 永远不会在内存中展开，这正是 GQA 的全部意义：`kv_head_idx` 的单个 K/V tile 被所有打包进 Q 行的 `GQA_RATIO` 个 query 头复用。输出端镜像输入端，用匹配的 3D 视图在 epilogue 后将打包的行存储回 `O[batch, seq, qo_head, dim]`。

结果是 GQA 完全存在于 Q 加载和 O 存储的边界上。在计算路径内部，分数 MMA 仍然看到一个普通的 `128 x HEAD_DIM` Q tile，其余 tile 原语图保持不变。

## Tile 调度

调度器的工作是映射每个 CTA 到一个 `(batch, kv_head, m_block)` attention 任务，而正确的策略取决于 masking 是否使这些任务在成本上均等：

- 非因果模式使用 `FlashAttentionLinearScheduler`。每个任务做相同的工作量，所以一个固定的 CTA 池按 `num_ctas` 推进就足以均匀分配它们。
- 因果模式使用 `FlashAttentionLPTScheduler`，因为因果 masking 使工作量极不均匀：靠近开始的 Q block 大致只 attention 一个 K/V block，而靠近结束的 Q block attention 所有 K/V block。朴素的拆分会导致一些 CTA 比其他的完成晚得多，因此最长处理时间调度器将重 block 前置以均匀完成时间，同时仍然将附近的 batch/head 任务保持在一起以获得 L2 局部性。

尽管有这些差异，两个调度器暴露相同的循环接口：

```python
while scheduler.valid():
    m_block_idx = scheduler.m_block_idx
    batch_idx = scheduler.batch_idx
    kv_head_idx = scheduler.head_idx
    # 处理一个 Q block 对其 K/V block 范围
    scheduler.next_tile()
```

唯一的行为差异在于 `next_tile()` 做什么：在非因果模式下，它将 CTA 推进到另一个任务，而在因果模式下，它在当前任务后结束循环。无论哪种方式，这纯粹是一个调度决定：它选择 CTA 拥有*哪个* attention tile，而不是该 tile 如何计算。在循环内部，无论调度如何，相同的本地原语都会运行：TMA 加载、分数 MMA、softmax、value MMA、修正、TMA 存储。

## 编译和验证

以上一切都是摘录，因此要将所有内容组合在一起并实际运行 kernel，我们从 `tirx-kernels` 导入真实的实现，编译它，并对 torch 参考进行验证。完整的 kernel（本章走过的每个部分都组装在一个文件中）是 `tirx-kernels` 仓库中的 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py)。有两件事与 GEMM 验证单元不同：Flash Attention 有一个更丰富的入口点（`get_flash_attention4_kernel`），并且它接受一个额外的 `profiler_buf` 参数用于其内置 profiler。这是为整章运行的唯一单元：

```python
import torch
import torch.nn.functional as F
import tvm
from tirx_kernels.attention.flash_attention4 import (
    get_flash_attention4_kernel, PROFILER_BUFFER_SIZE)

B, S, Hq, Hkv, D = 1, 1024, 32, 8, 128   # GQA：32 个 query 头共享 8 个 KV 头
Q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
O = torch.empty(B, S, Hq, D, dtype=torch.float16, device="cuda")
prof = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")

kernel = get_flash_attention4_kernel(B, S, S, Hq, Hkv, D, is_causal=False)
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
ex.mod(Q, K, V, O, prof)   # ex.mod 直接接受 torch tensor，与其他章节一样
torch.cuda.synchronize()

# torch 参考；enable_gqa 让 32 个 query 头共享 8 个 KV 头
qt, kt, vt = (x.transpose(1, 2).float() for x in (Q, K, V))
ref = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True).transpose(1, 2).half()
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)
print(f"FA4: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}, non-causal -> PASS")
```

**预期输出**：`... -> PASS`。kernel 在 fp32 中累积在线 softmax，然而几个不同的近似仍然将其结果与高精度参考分开。有输入和操作数的 fp16 存储和舍入；基于 `exp2` 的 softmax 重新表述（`scale_log2 = log2(e)/√d` 重新表述每个指数运算）；在线 softmax 重排序和逐行重缩放，以运行尺度而非一次性汇总 block；以及写回时 `O` 的 fp16 转换。此处选择的 `rtol`/`atol`（与源 kernel 自身测试使用的相同容差）的大小旨在覆盖所有这些相对于 torch 参考的差异，而不仅仅是 fp16 舍入本身。所以如果你在此处看到一个真正的失败，而不仅仅是边界上的近似，将其读作指向 softmax 路径的路标：一个被丢弃的 `s_ready` / `p_o_rescale` / `p_ready_2` 等待，或重缩放步骤未能应用的 `row_max` / `row_sum` 更新。这些正是本章花在 barrier 上的那些握手。

## 与 GEMM 的差异

下表沿着变化的轴比较了 FA4 和 GEMM：

| 方面 | GEMM | Flash Attention 4 |
|--------|------|-------------------|
| MMA 阶段 | 一个重复的 MMA | 分数 MMA 和 value MMA |
| MMA 之间的工作 | 除了流水线握手外没有 | 在线 softmax、masking 和 O 重缩放 |
| 运行状态 | 只有累加器 | row max、row sum、O 累加器 |
| 主要中间体 | 累加器 TMEM tile | S、P 和 O TMEM tile 区域 |
| Warp 角色 | TMA 生产者、MMA 消费者、写回 | TMA 加载、MMA、softmax、修正、TMA 存储 |
| Barrier | 主要是加载/计算/写回握手 | 额外的分数/softmax/value/修正握手 |
| 调度单元 | 输出矩阵 tile | attention 任务：`(batch, kv_head, m_block)` |

这些差异中的每一个都追溯到我们在本章开头提出的结构性变化：第二个 MMA，中间夹着 softmax。另一方面，底层的 TIRx 契约从未改变：

- tile 原语说明什么 tile 移动或计算，
- 周围的范围说明哪些线程协作，
- 布局说明 tile 在哪里，
- barrier 说明下一个角色何时可以消费它。

所以 FA4 比 GEMM 更难，不是因为它依赖于不同的硬件，而是因为简单地有更多的 tile 值和它们之间更多的握手。

## 练习

1. 与 GEMM 相比，FA4 中两个 MMA 阶段之间出现了什么新的 tile 握手？命名生产者、TMEM tile 和消费者。
2. 为什么 softmax 将分子 tile `P` 写回 TMEM 而不是只保留在寄存器中供 value MMA 使用？
3. 选择 `p_o_rescale` 或 `p_ready_2`。该 barrier 确切证明了什么？如果 value MMA 跳过该等待可能会出什么问题？

**与你的 agent 一起尝试**：选择一个未标注的 tile 原语，例如 epilogue 的 `Tx.copy_async`、fp32 -> fp16 的 `Tx.cast` 或第二个 `gemm_pv` 子 MMA。要求其范围/布局/分发/握手卡片，然后对照源守卫、分配和等待检查答案。
