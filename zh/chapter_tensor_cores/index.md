(chap_tensor_cores)=
# Tensor Core：`tcgen05`

:::{admonition} 概览
:class: overview

- `tcgen05` 是 Blackwell 的 Tensor Core 指令族。其 MMA 指令以协作方式执行数据块矩阵乘-累加工作，并由一个选定线程提交指令。
- 累加器位于 TMEM 而非寄存器中。尾声部分之后通过 `tcgen05.ld` 将其带回寄存器。
- `cta_group::1` 和 `cta_group::2` 控制由一个 CTA 还是两个 CTA 协作完成 MMA。这一选择也会改变 M 维度到 TMEM 的映射方式。
- 块缩放 MMA 模式（例如 `mxfp8` 和 `nvfp4`）增加了缩放因子操作数。数据操作数位于 SMEM 中，而缩放因子通过 TMEM 暂存。
:::

稠密线性代数是现代 GPU 完成绝大部分有用工作的地方。普通的 CUDA Core 矩阵乘法无法接近芯片的标称峰值（{ref}`chap_background`）。快速 GEMM 和注意力内核通过向 Tensor Core 提供正确形状、布局和同步方式的数据块来达到该峰值。

自 Volta 以来，基本操作在精神上一直没有改变。Tensor Core 消费矩阵数据块，将它们相乘，并累加结果。从一代到下一代发生变化的是操作的发起方式、操作数的布局以及累加器的存放位置。

Blackwell 对最后一部分做出了重大改变。`tcgen05` 的累加器不再作为长期存在的寄存器片段保留，而是被写入张量内存（Tensor Memory，简称 TMEM，{ref}`chap_tmem`）。这一改变影响了整个内核。MMA 写入 TMEM。完成状态被异步追踪。尾声部分稍后从 TMEM 加载累加器，并将其转换回所需的寄存器片段，用于类型转换和存储。

本章聚焦于计算指令本身。TMA（{ref}`chap_tma`）负责将操作数移入 SMEM。TMEM 负责存放累加器和部分缩放因子操作数。`tcgen05.mma` 是位于这两段内存移动之间的 Tensor Core 操作。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/tcgen05_intro.html" title="tcgen05 and Tensor Memory" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互操作：`tcgen05` 累加器行为。切换 A 或 B 的转置，选择输出宽度 `N`，并逐步执行 `K` 次迭代，观察部分和在 TMEM 中累积。*

## `tcgen05` MMA

`tcgen05` MMA 是 Blackwell 的 Tensor Core 矩阵乘-累加指令。它是一种协作指令。任务由一个 warpgroup 执行，在某些模式下还可以涉及来自同一集群的两个 CTA。该指令不是由每个线程独立发出的。一个选定的线程代表参与组提交操作。

将 MMA 拆解为三个问题有助于理解。

第一个问题是谁在协作。普通模式使用一个 CTA，写作 `cta_group::1`。更大的模式使用集群中的两个 CTA，写作 `cta_group::2`。在这两种情况下，指令代表的是对一个数据块的 Tensor Core 操作，而不是某个线程的标量操作。

第二个问题是操作数和结果位于何处。数据操作数通常位于 SMEM 中。某些变体也可以从 TMEM 读取 A 操作数。累加器写入 TMEM。操作数布局必须符合 Tensor Core 的期望，包括数据操作数所使用的 swizzled 共享内存布局（{ref}`chap_data_layout`）。

第三个问题是如何观察完成状态。`tcgen05.mma` 是异步的。发出 MMA 并不意味着乘-累加已经完成。指令在操作提交后返回，而 Tensor Core 继续运行。内核使用提交组（commit group）和 `mbarrier` 来获知结果何时就绪（{ref}`chap_async_barriers`）。

这种异步行为正是实现重叠的关键。一个快速的内核不会发出 MMA 后立即等待它完成。它可以发出 MMA，开始准备后续的数据块，仅在确实需要结果时才等待。代价是每一次交接都必须显式处理。如果尾声部分在 MMA 完成屏障触发之前读取 TMEM，那就会读得过早。

## Accumulator 位于 TMEM

在 Ampere 和 Hopper 上，累加器以寄存器形式暴露给程序。MMA 产生每线程寄存器片段，尾声部分直接消费该片段。这很简单，但它将累加器大小与每个线程的寄存器预算绑定在一起。

Blackwell 打破了这种联系。`tcgen05.mma` 将其累加器写入 TMEM——一个属于 CTA 作用域的 Blackwell 内存空间。累加器可以在整个计算阶段停留在 TMEM 中，尾声部分稍后使用 `tcgen05.ld` 将其加载回寄存器。

这改变了内核的形态。寄存器片段在边界处仍然重要。尾声部分仍然需要寄存器来执行转换、应用逐元素操作并存储结果。但是长期存在的累加器状态不再是一个寄存器分配问题，而是一个 TMEM 分配和布局问题（{ref}`chap_tmem`）。

这就是为什么 `tcgen05` 和 TMEM 必须放在一起理解。MMA 指令决定了计算哪个数据块。TMEM 决定了累加器的落位。尾声部分必须使用匹配的加载路径，以期望的寄存器布局恢复累加器。

## `cta_group::1` 和 `cta_group::2`

`tcgen05` MMA 可以在 `cta_group::1` 或 `cta_group::2` 模式下运行。

在 `cta_group::1` 模式下，一个 CTA 拥有该 MMA。其操作数位于该 CTA 的 SMEM 中，累加器写入该 CTA 的 TMEM 中。

在 `cta_group::2` 模式下，集群中的两个 CTA 协作完成一个 MMA 数据块。每个 CTA 拥有自己的 SMEM 和自己的 TMEM。累加器并非存储在一个跨越两个 CTA 的物理 TMEM 区域中，而是在两个 CTA 之间划分，每个 CTA 持有一部分。偶数 CTA 发出指令并提交这对 CTA 的完成屏障。

这一选择很重要，因为它改变了逻辑累加器数据块 `C(M, N)` 到 TMEM 的映射。TMEM 有 128 个硬件 Lane 行和多达 512 个硬件 Col 列。在 TIRx 布局记号中，这些轴写作 `TLane` 和 `TCol`。MMA 模式决定了 `C` 的行和列如何放置到这些 TMEM 轴上。

有四种值得关注的情况。

下面的图遵循演示中的颜色约定：紫色标记 SMEM 操作数，橙色标记 TMEM 累加器状态，绿色标记 Tensor Core MMA 路径。CTA 身份通过标签和位置来显示，而非改变这些硬件颜色。

### `cta_group::1`，`M = 128`

这是最简单的情况。一个 CTA 计算一个 128 行的数据块。TMEM 也有 128 个 Lane 行。因此映射是直接的：累加器的行 `m` 映射到 Lane `m`，N 维度映射到 TMEM 列。

结果占据 N 列上的 128 个 Lane 行。这是基线图景。CTA 在 SMEM 中拥有 A 和 B，并在其 TMEM 中拥有完整的累加器数据块。

![cta_group::1, M=128：行 m 直接映射到 TMEM Lane m](../img/mma_cg1_m128.svg)

### `cta_group::1`，`M = 64`

当 `M = 64` 时，累加器只有 64 行，但 TMEM 仍然有 128 个 Lane 行。硬件并非简单地将行 0 到 63 打包到 Lane 0 到 63。相反，它将它们以四段 16 行的方式分布在 128 个 Lane 上。

行 0 到 15 进入 Lane 0 到 15。行 16 到 31 进入 Lane 32 到 47。行 32 到 47 进入 Lane 64 到 79。行 48 到 63 进入 Lane 96 到 111。

这就在 Lane 16 到 31、48 到 63、80 到 95 以及 112 到 127 处留下了空隙。这些空隙是有意为之。使用不同的 Lane 对齐方式，另一个独立的 `M = 64` MMA 可以占用互补的 Lane。这使得两个较小的 M 数据块可以共享 128 行 TMEM 结构而不互相干扰。

N 维度仍然映射到 TMEM 列。不寻常的部分仅仅是 M 行在 Lane 上的放置方式。

![cta_group::1, M=64：四个 16 行段，Lane 步长为 32，为另一个对齐的 M=64 数据块留有空间](../img/mma_cg1_m64.svg)

### `cta_group::2`，`M = 256`

当 M 维度大于一个 CTA 自然能容纳的大小时，MMA 可以使用 `cta_group::2`。对于 `M = 256`，划分是直接的。CTA 0 持有行 0 到 127。CTA 1 持有行 128 到 255。

每个 CTA 使用自己的 TMEM Lane 行 0 到 127 和完整的 N 列。物理上，这是两个独立的 128 行 TMEM 区域，每个 CTA 一个。逻辑上，它们形成一个 256×N 的累加器数据块。

每个 CTA 还提供与其 M 行对应的 A 部分。根据模式要求，B 对两个 CTA 都可用。偶数 CTA 负责发出 MMA 并为这对 CTA 提交完成屏障。

这就是 {ref}`chap_gemm_advanced` 中两 CTA 集群 GEMM 所使用的模式。

![cta_group::2, M=256：M 在两个 CTA 之间连续划分，每个 CTA 128 行](../img/mma_cg2_m256.svg)

### `cta_group::2`，`M = 128`

`cta_group::2`、`M = 128` 模式仍然使用两个 CTA，但 M 维度更短。由于总共只有 128 行，每个 CTA 获得 64 个 M 行。

剩余的 Lane 容量用于打包 N 维度。在每个 CTA 内部，N 的一半占用 Lane 0 到 63，另一半占用 Lane 64 到 127。这使得每个 CTA 即使只拥有 64 个 M 行，也能使用全部 128 个 Lane 行。

因此划分包含两个部分。M 在 CTA 对之间划分，每个 CTA 64 行。然后 N 在每个 CTA 内部，在 TMEM Lane 行的低半部分和高半部分之间划分。

![cta_group::2, M=128：每个 CTA 64 个 M 行，N 的两半在 Lane 低半部分和高半部分之间堆叠](../img/mma_cg2_m128.svg)

在这些模式下，原则是相同的。`tcgen05.mma` 计算一个逻辑累加器数据块，但该数据块必须被放置到物理的 128 Lane × 最多 512 Col 的 TMEM 空间中。模式和 M 形状决定了这种放置。内核的其余部分在稍后读回累加器时必须使用相同的映射。

对于本文中的内核，累加器通常采用 TMEM 中的 f32。这是常见的高精度路径。但它不是唯一的累加器类型。`.kind::f16` 路径可以以 f16 累加。

## Operand Placement

对于稠密 MMA 模式，A 和 B 在 MMA 运行之前在 SMEM 中准备就绪。TMA 负责将全局内存数据块移入 SMEM。内核以 Tensor Core 所期望的布局（包括任何所需的 swizzle）来安排这些 SMEM 数据块。

累加器 C 写入 TMEM。这是与早期世代的主要区别。尾声部分不会直接以 MMA 指令的输出形式收到累加器。它必须显式地使用 `tcgen05.ld` 从 TMEM 加载。

在 `cta_group::1` 中，一个 CTA 提供操作数并拥有累加器。在 `cta_group::2` 中，每个 CTA 从其自己的 SMEM 提供自己一侧的操作数，并且每个 CTA 拥有累加器的一部分 TMEM。当 A 按 M 划分时，每个 CTA 保留其自己 M 分片的 A 行。B 根据模式共享，因为两个 M 分片都与同一个 N×K 数据块相乘。

在阅读内核时，这种分离很重要。SMEM 放置回答了 Tensor Core 如何读取 A 和 B。TMEM 放置回答了累加器去哪里。这两种布局由 MMA 模式关联，但它们不是同一个内存空间，不能互换处理。

## Block-Scaled MMA

稠密模式直接从 SMEM 读取其数据操作数并累加到 TMEM。块缩放 MMA 增加了两个额外操作数：A 和 B 的缩放因子张量。

这用于非常低精度的格式，例如 `mxfp8` 和 `nvfp4`。低精度格式效率高，但它们的动态范围很小。单一的全局缩放通常过于粗糙。如果缩放针对最大值选择，较小的值会损失精度。如果缩放针对小值选择，较大的值可能会被截断。

块缩放通过为小的 K 块分配缩放因子来解决这个问题。一组连续的 K 元素共享一个缩放。MMA 在概念上用各自的缩放对每个块进行反量化，然后以累加器类型累加乘积。

对于 A 和 B，这引入了两个缩放因子张量：

```text
SFA(M, SFK)
SFB(N, SFK)
```

其中 `SFK = K / B`，`B` 是沿 K 方向的块大小。

确切的块大小取决于格式。重要的是缩放轴以更粗的粒度跟随 K。每个缩放因子描述一个 K 值块，而不是单个元素，也不是整个矩阵。

数学形式为：

```text
acc += (Aq * scale_a) * (Bq * scale_b)
```

其中 `Aq` 和 `Bq` 是量化后的低精度值，缩放因子在累加之前恢复它们的大致大小。

缩放的数据类型也很重要。使用 `e8m0` 缩放时，每个缩放实际上是 2 的幂。使用 `e4m3` 缩放时（如 `nvfp4` 使用的），缩放的浮点值较小，可以表示 2 的幂之间的值。

## Scale Factor 放在哪里

块缩放 `tcgen05.mma` 在一条重要的放置规则上与稠密 MMA 不同：缩放因子从 TMEM 读取。

数据操作数 A 和 B 仍然暂存在 SMEM 中。缩放因子 SFA 和 SFB 通过 TMEM 暂存。由于 TMA 加载到 SMEM，缩放因子通常需要额外一步。内核首先将它们加载到 SMEM，然后通过 `tcgen05.cp` 从 SMEM 拷贝到 TMEM。只有当缩放因子位于 TMEM 中之后，块缩放 MMA 才能读取它们。

这使得缩放因子与数据操作数有不同的移动路径：

```text
A, B：     全局内存到 SMEM，然后 MMA 读取 SMEM
SFA, SFB： 全局内存到 SMEM，然后 tcgen05.cp 将 SMEM 拷贝到 TMEM，然后 MMA 读取 TMEM
```

缩放因子的 TMEM 布局是紧凑的。一个 128 行的缩放向量可以打包到 32 个 Lane 行中，使用基于 `r % 32` 的 Lane 位置和 `r / 32` 的列映射。数据随后可以被广播到读取完整 128 个 Lane 空间的四个 warp 上（{ref}`chap_layout_generations`）。

这是一个很好的例子，说明为什么 TMEM 布局必须显式表达。累加器布局和缩放因子布局都在 TMEM 中，但它们不是相同的布局。累加器使用 MMA 的输出映射。缩放因子使用块缩放 MMA 所期望的紧凑布局。

## `cta_group::2` 中的 Scale Factor

在两 CTA 的情况下，缩放因子跟随它们所缩放的数据。

SFA 缩放 A。由于 A 按 M 在 CTA 对之间划分，SFA 也按 M 划分。每个 CTA 持有与其 A 行相对应的 SFA 行。

SFB 缩放 B。由于两个 CTA 都与同一个 B 数据块相乘，SFB 必须对两个 CTA 都可见。在实践中，这意味着 SFB 被多播到 CTA 对。

这就是块缩放集群 GEMM 中常见加载模式的来源。SFA 按每个 CTA 加载，使用该 CTA 自身 M 分片的掩码。SFB 广播到两个 CTA，因为两个 CTA 都需要相同的 N 侧缩放因子。

![块缩放 MMA 放置：A 和 B 打包在 SMEM 中；SFA、SFB 和 C 在 TMEM 中，其中 SFA 按 M 在 CTA 之间划分，SFB 在 CTA 对之间多播](../img/mma_block_scaled.svg)

## 保持 MMA Contract 匹配

一个 Blackwell GEMM 数据块经过多个专门的路径。

TMA 将 A 和 B 从全局内存带入 SMEM。对于块缩放模式，它还将缩放因子带入 SMEM。`tcgen05.cp` 在需要时将那些缩放因子移入 TMEM。`tcgen05.mma` 读取其操作数，在 Tensor Core 上异步运行，并累加到 TMEM。完成屏障告知内核累加器何时就绪。尾声部分随后使用 `tcgen05.ld` 将累加器从 TMEM 加载回寄存器，并存储最终输出。

在这些路径中，内核必须保持三个约定匹配：SMEM 操作数布局、TMEM 累加器或缩放因子布局，以及使下一个消费者可以安全运行的异步完成信号。
