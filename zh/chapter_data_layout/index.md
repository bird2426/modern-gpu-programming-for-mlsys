(chap_data_layout)=
# 数据布局及其记号

:::{admonition} 概览
:class: overview

- **数据布局**将张量的逻辑索引映射到物理位置，它决定了合并访问、存储体冲突，以及某个引擎能否读取一个数据块。
- 本书使用统一的记号描述布局：`S[(shape) : (strides)]`，配合命名轴（`@laneid`、`@TLane`……）以及一个表示广播或复制数据的复制项 `R[...]`。
- Swizzle 是一种对地址进行 XOR 重映射的技术，用于消除共享存储体冲突。
:::

相同的数字，若以不同的物理排列写入内存，在同一块 GPU 上的运行速度可能相差一个数量级。

原因在于，张量的逻辑索引完全无法说明其字节实际位于何处。硬件对此布局高度敏感：它决定了 32 条线程的加载是合并为一次事务，还是分散为 32 次；决定了这些地址是落在不同的存储体上，还是冲突并串行化；甚至决定了某个数据块的字节排列是否能够被 Tensor Core 所读取。

机器学习程序通常用逻辑形状来描述张量。而**数据布局**则补充了缺失的物理部分：它指明了逻辑索引 `(i, j, …)` 对应的元素位于何处——无论在内存中、寄存器中，还是其他某种硬件存储中。

本章将介绍现代 GPU 编程中几类主要的布局。为了使讨论易于处理，我们将开发一套紧凑的**记号**，用以统一描述机器学习系统可能遇到的各类情形。最后，我们以 **swizzling** 作为收尾——这种机制能让对同一个数据块的按行访问和按列访问同时保持高效。

## Shape–Stride 模型

在讨论 GPU 特有的布局之前，值得从最简模型开始，因为本章后续的所有内容都建立在此之上。布局的核心只有两样东西：**形状**（shape）和与之匹配的一组**步长**（strides）。我们将这一对记为 `S[(shape) : (strides)]`，要查找某个逻辑索引对应的位置，只需将该索引与步长做点积。例如，一个行主序的 4×4 矩阵如下所示：

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1
```

这不过是经典的 shape/stride 模型，以一种紧凑的方式书写（CuTe 记法的行主序简化版），后续所有内容都由此构建。

事实上，你几乎肯定已经使用过这个模型了。任何写过 PyTorch 或 NumPy 的人都用过，因为那些库中的张量*本质上*就是形状和在一段连续存储缓冲上的步长：

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← 正是 S[(3, 4) : (4, 1)]
```

当你以这种方式看待张量时，就会明白为什么那么多"重塑"操作根本无需触碰数据。它们只是改写步长，然后返回同一段存储上的一个**视图**。最清晰的例子就是转置（transpose）或 permute：

```python
tt = t.permute(1, 0)               # 或 t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← 步长互换，未移动数据
tt.data_ptr() == t.data_ptr()      # True，同一段字节
```

这里 `t.permute(1, 0)` 在*同一块*内存上是 `S[(4, 3) : (1, 4)]`：转置纯粹是步长的变化，没有一个字节被移动。对于连续张量的 `reshape` 或 `view` 也是如此：在旧存储之上使用新的形状和步长。（NumPy 的行为完全相同；唯一的区别在于它的 `.strides` 以字节而非元素为单位计数。）

这正是布局在 GPU 上的工作方式，本章其余部分实际上是对同一个核心思想的一系列变奏：数据块的映射（无论是映射到内存，还是通过我们即将介绍的命名轴映射到线程和寄存器）都是固定缓冲上的步长规则，因此重排数据块通常只是*布局*的改变，而非拷贝。不过，我们需要对这种推理的边界保持谨慎。零拷贝的成立条件是逻辑视图建立在单个线性地址空间之上；在 GPU 上，只有当新视图与现有的字节和所有权安排兼容时才成立。一旦需要改变某个元素所属的线程或寄存器，或者改变 SMEM 的 swizzle，通常都需要真正的数据移动：加载、存储、混洗（shuffle）、`ldmatrix`、转置。

## Tile Layout

到目前为止，我们描述的都是整个张量的布局。然而，GPU 内核很少一次操作整个矩阵；它们通常处理较小的数据块（tile），这些数据块由不同的硬件部件加载、变换并执行计算。好消息是，分块并不需要引入新的概念。它仍然只是一种布局，只是现在多了几个维度。将 8×8 矩阵切分为 2×4 的数据块，就得到了一个 4D 布局，坐标为 `(tile_row, row_in_tile, tile_col, col_in_tile)`，并且步长的选择使得每个数据块保持连续：

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

一个逻辑索引 `(i, j)` 首先变为 `(i//2, i%2, j//4, j%4)`，然后经步长计算得到地址。值得注意的一点是，这种记号在不借助任何特殊的"tile"概念的情况下就表达了分块：它与之前的 shape–stride 模型完全相同，只是索引被拆分为外部坐标和内部坐标而已。

下面的交互式可视化展示了逻辑矩阵索引如何被分解为 tile 坐标，然后映射到物理地址。

```{raw} html
<iframe src="../demo_zh/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互操作：点击某个单元格，查看其分块索引和地址。*

## 命名轴

到目前为止，`S[...]` 中的每个步长都表示线性内存中的一个偏移量，我们将地址视为该空间中的某个位置。然而，在 GPU 上，数据可以存在于多个地方：除了内存，一个数据块还可能分布在 warp 线程上、线程寄存器上，或者 TMEM 的 Lane 和 Col 上。为了统一描述所有这些情况，我们对记号进行扩展，引入**命名轴**（named axes）。其思想是让每个步长系数携带一个轴标签，指明它行进在哪个空间中：`@m` 表示普通内存，`@laneid` 表示 warp 线程，`@reg` 表示寄存器，`@warpid` 表示 warp，而 `@TLane` / `@TCol` 表示 TMEM 坐标。有了这些标签，单一的布局不仅可以描述数据在内存中的位置，还能描述它在操作该数据的硬件资源上的分布情况。

将内存标签显式化之后，内存中一个行主序的 8×16 数据块就简单地表示为：

```text
S[(8, 16) : (16@m, 1@m)]
```

标签的真正价值体现在布局描述的是**分布在多个线程上**的数据（而非在内存中连续排列）。例如 `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]`：它不再指向线性内存，而是将行和列映射到线程 ID 和每个线程的寄存器。这里 `laneid` 表示 warp 内部的线程索引，即 `thread_index % warp_size`。这正是在 {ref}`chap_layout_generations` 中将要遇到的 Tensor Core 寄存器片段。

下面的交互式可视化展示了布局如何将张量元素分布到 warp 线程和每线程寄存器上，而非放置在线性内存中。

```{raw} html
<iframe src="../demo_zh/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互操作：一个基于 `@laneid` 和 `@reg` 的布局；点击某个单元格，查看它由哪个 lane/register 持有。*

## 分布式布局

命名轴如此有用的原因在于，它让我们能够统一描述系统中多个层级上的放置方式，包括跨*整个设备*的放置方式。刚才我们仅在单个 GPU 内部使用它们来描述线程和寄存器，但同样的思想可以向外延伸：诸如 `@gpuid_x` 和 `@gpuid_y` 这样的轴可以表示数据在 GPU 网格中的位置，借助它们，该记号就能捕捉分布式训练和推理中出现的切分模式。然而，这些轴尚未能描述*复制*（数据被拷贝到多个位置），因此我们增加了记号 `R[n : stride]`，其中 `R` 表示被复制的维度。例如，`R[2 : 1@gpuid_x]` 表示沿 `@gpuid_x` 轴的复制。将两者结合，一个单一的表达式既可以跨 2×2 GPU 网格切分张量，又可以沿某一轴复制：

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

下面的演示在一个小型 GPU 网格上展示了这种分区与复制相结合的模式。点击任意单元格查看它由哪个设备持有，并观察 `@gpuid_x` 复制如何将相同的副本放置在配对的设备上；按钮可在完全切分、切分+复制和切分+偏移三种布局之间切换。

```{raw} html
<iframe src="../demo_zh/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互操作：一个分布在 2×2 GPU 网格上的布局；点击某个单元格，查看它由哪些设备持有。*

### Kernel 内复制模式：TMEM 中的 Scale Factor

我们刚才为 GPU 网格引入的复制维度 `R[...]` 并不仅限于多个设备。事实表明，同一结构也可以描述完全发生在单个 kernel 内部的现象：即硬件**跨线程广播**的数据。Blackwell 的块缩放 MMA（{ref}`chap_layout_generations`）就是一个很好的例子。其缩放因子（scale factors）存在于 TMEM 中，一个 128 行的缩放向量仅存储在 **32 个 TMEM Lane** 中，逻辑行号 `r` 映射到 TMEM Lane `r % 32`，而 `r // 32` 沿列方向排列。这 32 个已存储的 TMEM Lane 随后沿 TMEM 的 `TLane` 轴**复制**，从 32 个 TMEM Lane 扩展到 128 个 TMEM Lane，使得读取 warpgroup 中的四个 warp 各自在其 32-Lane 的 TMEM 窗口内找到一份副本。这就是 `warpx4` 广播，我们使用复制维度来描述它。读取操作由这些 warp 中的线程执行：

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

这样就在 TMEM Lane 步长为 32 的位置上生成了四个副本：TMEM Lane `l`、`l+32`、`l+64` 和 `l+96` 都持有相同的缩放值。与之前一样，复制维度并不携带新数据；它只是说"同一个值，位于四个 TMEM-Lane 位置上"，正如刚才 `@gpuid_x` 将一行数据广播到 GPU 网格那样。

下面的交互式演示将两个步骤一并展示：首先紧凑地打包到 32 个 TMEM Lane 中，然后通过 `warpx4` 广播扩展到 128 个读取 Lane。

```{raw} html
<iframe src="../demo_zh/sf_tmem.html" title="Scale factors in TMEM: packing and warpx4 replication" loading="lazy"
        style="width:100%; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互操作：点击一个缩放因子 `SFA[m, sf]`；它先被打包到 TMEM Lane `m mod 32`、列 `(m // 32)·4 + sf` 处，然后通过 `warpx4` 沿 `TLane` 轴广播到四个 Lane 副本（`l`、`l+32`、`l+64`、`l+96`），每个副本位于一个 warp 的 32-Lane 窗口中。*

每列内部的字节打包方式（`scale_vec` 的 1X/2X/4X 模式）以及 `cta_group::2` 的划分将在 {ref}`chap_layout_generations` 中介绍。

已经了解 CuTe 的读者可以将本章的记号视为 CuTe 的行主序变体，并扩展了显式的硬件命名轴和专用的复制结构。

## Swizzle Layout

本章的最后一个布局是为了解决一个特定的硬件问题。GPU 上的共享内存被组织成多个存储体（bank），当不同线程落在不同存储体上时，访问速度最快。而当多个线程访问*同一*存储体中的不同地址时，硬件只能串行化处理，我们就需要付出**存储体冲突**（bank conflict）的代价。

在张量计算程序中，这几乎是无法避免的，因为内存并不是以纯线性的顺序被访问。在处理矩阵时，我们经常需要读取同一数据块的行切片和列切片，这就产生了一个真正的矛盾：对行访问高效的布局往往会在列访问时产生存储体冲突，而对列友好的布局则会损害行访问。**Swizzling** 正是用来打破这一矛盾的技术。

Swizzle 背后的思想是对地址映射进行排列（通常通过将列索引与行索引进行 XOR 运算），从而使*行和列*的访问都能均匀分布在存储体上。它所提供的无冲突保证是有特定范围的：仅适用于匹配的元素宽度、swizzle 模式以及访问模式（即引擎描述符所期望的模式），而不适用于任意元素宽度或对齐方式。

下面的第一个交互式演示将这一点具体化了。点击一个列索引，观察每个元素落在哪个存储体上：左侧的普通行主序数据块中，某一列的所有八个元素都落入同一个存储体，因此读取被串行化为八个周期；而在右侧经过 XOR-swizzle 的布局中，同一列被分散到八个不同的存储体上，只需一个周期即可读取完毕。

```{raw} html
<iframe src="../demo_zh/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互操作：一个 8×8 数据块，在普通行主序中按列访问会产生存储体冲突，经过 XOR swizzle 后则无冲突。*

这个小小的 8×8 示例展示了核心思想，但实际的 GPU 内存拥有的存储体远不止那个玩具图所暗示的那么多。要使 swizzling 在完整规模下正常工作，我们不会将整个数据块当作一个整体对象处理。相反，我们将内存切割成小段，并在每个段内部应用 swizzle 模式。实践中最常见的情况是 `SWIZZLE_128B`，它围绕 128 字节段组织，使得行/列重映射的技巧自然地适配 32 存储体的内存系统。

下面的交互式演示展示了具体的硬件 swizzle `SWIZZLE_128B`，在我们将各种格式泛化之前，可以直观地看到逐段重复的模式。

```{raw} html
<iframe src="../demo_zh/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互操作：128 字节段内的 `SWIZZLE_128B` 模式；逐步执行读取周期，观察 `physical_sector = logical_sector XOR row` 如何将每一列分散到不同的存储体上。*

同样的思想也适用于 128 字节以外的情形。为了简化可视化，我们将使用单一色块来表示一个段，而不是绘制每个独立的存储体。通常，硬件会定义一个重复的**原子**（atom），对其施加排列变换，不同的 swizzle 模式会选择不同的原子大小。`SWIZZLE_128B` 使用 8 × 128 B 的原子，`SWIZZLE_64B` 使用 8 × 64 B 的原子，`SWIZZLE_32B` 使用 8 × 32 B 的原子；整个数据块由所使用的原子平铺而成。

最后一个交互式演示允许你在这些格式之间切换（包括一种 16 B 交错模式），选择数据类型，并悬停在任意单元格上直接查看一个原子内部的元素排列——这是推理某条加载/存储指令期望何种 swizzle 时所需要的信息粒度。

```{raw} html
<iframe src="../demo_zh/swizzle_atom_general.html" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互操作：选择一种 swizzle 格式（和数据类型），查看其原子形状（8 × N B）；悬停在一个单元格上，查看其元素是如何排列的。*

那么应该选择哪种模式呢？经验法则是优先选择数据块能填满的*最大*原子。N 字节原子需要数据块的连续维度至少为 N 字节且是其倍数，因此 `SWIZZLE_128B` 仅当一行至少跨越 128 字节（即 64 个 `float16` 元素）时才适用。当条件满足时，它是默认选择，因为其 8 × 128 B 的原子覆盖了完整的 128 字节存储体行，从而一次将一列分散到全部 32 个存储体上，使得 fp16 下每次可无冲突地访问 8 行和 8 列。但是，当问题形状迫使连续维度较小时，数据块无法填满 128 B 原子，就需要降级到 `SWIZZLE_64B` 或 `SWIZZLE_32B`，即该行仍能覆盖的最大原子。

你永远不会手工计算这些排列后的地址，而有必要精确说明 swizzle 与 `S[...]` 记号之间的关系：它**不属于**那个仿射映射。它是一个独立的、非仿射的层，叠加在仿射映射之上。`S[...]` 布局将元素放置在线性内存（`@m`）地址上，然后 swizzle 对该地址进行排列，在 TIRx 布局 API 中写作 `ComposeLayout(swizzle, tile)`（{ref}`chap_tirx_layout_api`）。你只需为所有操作该数据块的指令选择一致的 swizzle 模式，组合后的布局会处理其余部分。

这个组合后的布局正是硬件所填充的，也是 swizzling 和分块相结合的地方。TMA 描述符是多维的，因此单个三维盒子既能描述数据块的原子平铺方式，也能描述每个原子内部的 swizzle 方式；一次 TMA 加载按原子逐个布局数据块，并在写入共享内存时进行 swizzle（{ref}`chap_tma`），无需单独的 swizzle 步骤。*每个引擎要求哪种 swizzle 模式是各代 GPU 特有的，这正是下一章的主题。*
