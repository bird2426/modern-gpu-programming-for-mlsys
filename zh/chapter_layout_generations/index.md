(chap_layout_generations)=
# 跨 GPU 世代的 Tensor Core 操作数布局

:::{admonition} 概览
:class: overview

- 在 Ampere、Hopper 和 Blackwell 上，Tensor Core 执行的高级操作仍然是：`D = A B + C`。
- 每一代之间发生变化的是操作数如何到达 Tensor Core、支持哪些数据块形状和数据类型，以及累加器位于何处。
- Ampere 使用 warp 级寄存器片段。共享内存数据块通过 `ldmatrix` 加载到片段中，累加器保留在寄存器中。
- Hopper 允许 `wgmma` 通过矩阵描述符直接从共享内存读取操作数。该描述符指明了 Tensor Core 所期望的共享内存 swizzle 格式。
- Blackwell 保留了共享内存操作数路径，但将累加器移到了 TMEM 中。块缩放 MMA 也将其缩放因子存放在 TMEM 中。
- 在所有世代中，有两个内存约束始终存在：全局内存合并访问和共享内存存储体冲突。
:::

从远处看，Tensor Core 的操作似乎很稳定。它将 A 和 B 的数据块相乘，加上累加器 C，然后产生 D。自 Volta 以来，这种形式一直保持不变。

但围绕该操作的细节并非一成不变。在一代 GPU 上快速运行的内核，在下一代上可能很慢。使用了错误布局的内核甚至可能计算出错误的结果，即使逻辑上的数学运算仍然是 `D = A B + C`。原因在于 Tensor Core 消费的不是抽象的矩阵，而是以非常具体的硬件布局消费操作数。

本章将追踪三代 GPU 上的布局约定。Ampere 通过 warp 级寄存器片段暴露 Tensor Core。Hopper 将输入操作数移到共享内存描述符中。Blackwell 保留了共享内存操作数，但将累加器移到了 TMEM 中。操作仍然是矩阵乘-累加，但进入和离开 Tensor Core 的路径每次都发生了变化。

来自 {ref}`数据布局 <chap_data_layout>` 章的布局记号是我们描述这些约定的语言。Blackwell TMEM 的细节将在 {ref}`chap_tmem` 中单独介绍。

## 两个始终存在的约束

在涉及 Tensor Core 之前，两个普通的内存约束已经在影响 GPU 内核的布局。

第一个是全局内存合并访问。当一个 warp 的 32 条线程发出全局内存加载时，内存系统希望这些地址落在少数连续且对齐的内存段中。如果地址是分散的，warp 加载就会变成多个内存事务。同样的逻辑数据移动需要更多带宽和时间。

第二个是共享内存存储体冲突。共享内存被划分为 32 个存储体。如果 warp 中的线程访问映射到同一存储体的不同地址，这些访问无法同时被服务。硬件会将它们串行化。一个在扁平共享内存数组看来无害的布局，因此可能由于其存储体模式而变得缓慢。

Swizzling 是解决共享内存一侧问题的常用方法。逻辑数据块保持不变，但物理地址映射被重新排列，使得访问模式分散到各个存储体，而不是堆积到同一个存储体上。

这两个约束甚至适用于从不使用 Tensor Core 的内核。Tensor Core 内核增加了第三个约束：操作数必须按照 Tensor Core 指令本身所期望的布局排列。本章的其余部分讨论这个第三约束在 Ampere、Hopper 和 Blackwell 三代之间的变化。

## Ampere：Warp Lane 上的寄存器 Fragment

在 Ampere 级别的 GPU 上，主要的 Tensor Core 指令是 warp 级的 `mma.sync.aligned.m16n8k*` 系列。重要的事实在于指令读写数据的位置：寄存器。

A、B 以及 C 或 D 累加器都是分布在 warp 的 32 条线程上的每线程寄存器片段。共享内存只是暂存区。在 MMA 运行之前，操作数数据块必须从共享内存移动到指令所期望的确切寄存器片段布局中。

数据路径如下所示：

```text
SMEM 到寄存器，使用 ldmatrix
寄存器到寄存器，使用 mma.sync
寄存器回到 SMEM，使用普通 store
```

Ampere 布局的故事大部分都遵循这条路径。内核必须将数据块以可高效加载的形式存储在共享内存中，然后使用 `ldmatrix` 产生 `mma.sync` 所期望的寄存器片段。

## Ampere Tensor Core 期望的输入

Ampere Tensor Core 读取由 8×8 子块单元构建的寄存器片段。这些就是 `ldmatrix` 加载并且 MMA 所消费的单元。

以 fp16 或 bf16 输入和 fp32 累加的 `mma.m16n8k16` 为例。累加器数据块的形状为 `16×8`。它以固定的模式分布在 32 条线程上。

对于 C 或 D 累加器，线程 `l` 持有行：

```text
l / 4
l / 4 + 8
```

和列：

```text
2 * (l % 4)
2 * (l % 4) + 1
```

因此每条线程拥有四个 fp32 累加值：来自两个 8 行半块的两行，与两个相邻列交叉。连续四条线程覆盖一行中的八列。

A 操作数使用相同的 M 侧行划分。K 维度分布在 `l % 4` 和线程所持有的寄存器上。对于 fp16 或 bf16，每个 32 位寄存器打包两个 K 值。

B 操作数使用匹配的 K 放置，并将 N 侧分布在线程组和寄存器上。

确切的细节因指令形状和数据类型而异，但原则是固定的。Tensor Core 期望一个特定的每线程寄存器片段。如果数值不在那些寄存器中且不满足该模式，指令将会得到错误的乘法结果。

在布局记号中，m8n8 片段是使用命名线程轴描述的模式，例如：

```text
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@m)]
```

两个 `laneid` 维度一起描述了行和列片段如何分布在线程上，而最后的 `m` 分量描述了每线程寄存器槽。

## `ldmatrix`：从共享内存到寄存器 Fragment

`ldmatrix` 是 Ampere 上连接共享内存和 Tensor Core 寄存器片段的指令。它是一个 warp 集合加载。一条指令将一个或多个 8×8 的 16 位矩阵从共享内存加载到 `mma.sync` 所期望的分布式寄存器布局中。

指令形式为：

```text
ldmatrix.sync.aligned.m8n8.x1.shared.b16
ldmatrix.sync.aligned.m8n8.x2.shared.b16
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

带有一个可选的 `.trans` 限定符。

`.x1`、`.x2` 和 `.x4` 形式分别加载一个、两个或四个 8×8 矩阵。行基地址由各个线程提供。对于矩阵 `m` 和行 `r`，基地址来自线程 `m * 8 + r`。这意味着 `.x1` 使用线程 0 到 7 提供行地址，`.x2` 使用线程 0 到 15，`.x4` 使用线程 0 到 31。

结果直接写入 MMA 片段。对于基本的 8×8 情况，线程 `l` 接收 Tensor Core 所期望的行和列对。使用普通循环执行每线程 `ld.shared` 指令需要手动重现这种散布。`ldmatrix` 将共享内存到片段的重新排列作为一条 warp 集合指令完成。

`.trans` 形式在加载时对每个 8×8 矩阵进行转置。当操作数存储的方向与 MMA 指令所期望的方向相反时使用。

![ldmatrix 将共享内存中的 8x8 数据块加载到 warp 寄存器片段中；在 Ampere 上反向路径使用普通 store，专用的 stmatrix 指令后来出现在 Hopper 上](../img/ldstmatrix.svg)

## 写回 Ampere Fragment

在 `mma.sync` 完成后，累加器仍然是一个寄存器片段。内核的尾声部分（epilogue）必须将该片段移出。

在 Ampere 上，没有与 `ldmatrix` 对应的专用反向指令。内核使用普通的每线程 store（有时在 store 之前配合 warp shuffle 或局部重排）将累加器以有用的布局写入共享内存或全局内存。

这使得 Ampere 模型简单，但也将大量的布局工作暴露给内核。输入端使用 `ldmatrix` 创建片段。计算指令读写寄存器片段。输出端则通过那些片段上的普通 store 处理。

## Ampere 上的 Swizzle

Ampere 内核已经需要使用共享内存 swizzle。原因在于共享内存数据块通常以一种访问模式写入，而以另一种模式读取。

假设一个数据块从全局内存沿行方向填充。行主序布局使得这种写入是合并的且对存储体友好。但是 `ldmatrix` 后面可能会以一种实际上沿列方向或跨越 8×8 子块的方式读取该数据块。对于普通的行主序布局，这些读取可能堆积到同一共享内存存储体上。

对于一个简单的 `(8, 64)` float16 数据块，一行是：

```text
64 * 2 bytes = 128 bytes
```

这正好是一整条共享内存存储体行。沿固定列向下移动，每行前进 128 字节，因此存储体索引会重复。八行可能折叠到同一个存储体上，产生 8 路冲突。

改为普通的列主序布局并不能完全解决问题。它通常会将冲突转移到另一方。行写入变得更差，而列式读取变得更好。

XOR swizzle 通过使物理列依赖于行来解决这个问题。一个简单的版本是：

```text
physical_col = logical_col xor row
```

逻辑数据块保持不变。共享内存中的物理布局被重新排列，使得行式写入和 Tensor Core 读取模式都能避免存储体冲突。

在 Ampere 上，这种 swizzle 通常通过手写的共享内存索引计算来表达。后来的世代将其变为硬件引擎使用的描述符格式的一部分。

![在普通行主序数据块上，行写入分散到多个存储体，而列读取则碰撞到同一个存储体上；XOR swizzle 将列读取分散到各存储体，同时不牺牲合并的行写入](../img/swizzle_conflict.svg)

## Hopper：`wgmma`、共享内存 Descriptor 和 Swizzle Format

Hopper 改变了 Tensor Core 路径的输入侧。它不再要求每个操作数都通过 `ldmatrix` 加载到寄存器中，Hopper 的 `wgmma` 可以直接从共享内存读取操作数。

B 操作数从共享内存矩阵描述符中读取。A 操作数可以从共享内存描述符或寄存器中读取，分别对应 `.ss` 和 `.rs` 形式。

这移除了从 SMEM 来源的操作数的显式 `ldmatrix` 步骤。但它并没有移除布局要求。Tensor Core 仍然期望操作数以精确的共享内存格式存储。不同之处在于，该格式现在通过矩阵描述符描述给硬件。

## Hopper Tensor Core 期望的输入

Hopper 共享内存矩阵描述符是对共享内存中矩阵数据块的紧凑描述。它告诉 `wgmma` 如何将逻辑操作数坐标转换为共享内存地址。

该描述符包含以下字段：

```text
起始地址 (start address)
主导维度偏移 (leading dimension offset)
步进维度偏移 (stride dimension offset)
swizzle 模式 (swizzle mode)
基址偏移 (base offset)
```

具体的解释取决于操作数的主序模式。对于 K-major 数据块，一个步长沿 K 方向前进，另一个沿 M 方向前进。对于 MN-major 数据块，角色互换。

Swizzle 模式是共享内存描述符格式之一，例如：

```text
SWIZZLE_NONE
SWIZZLE_32B
SWIZZLE_64B
SWIZZLE_128B
```

Swizzle 模式决定了两件事。它决定了描述符所使用的原子（atom）形状，也决定了在该原子内部应用的 XOR 排列。例如，128 字节 swizzle 模式将操作数视为一个由 8 行 × 128 字节的原子组成的网格，swizzle 在每个原子内部应用。

内核仍然需要正确放置字节。TMA 通常填充共享内存数据块，并且 TMA 描述符必须使用与 `wgmma` 描述符随后命名相同的 swizzle 格式。如果 TMA 写入的是 128 字节 swizzle 的数据块，`wgmma` 描述符就必须将其作为 128 字节 swizzle 的数据块来读取。如果描述符和数据不一致，Tensor Core 将读到错乱的操作数。

这是与 Ampere 的主要区别。Swizzle 不再仅仅隐藏在手写的共享内存索引中。Hopper 将其提升为第一级的描述符格式。写入数据块的 TMA 加载和读取数据块的 `wgmma` 指令都可以命名相同的格式。

![Hopper 共享内存矩阵描述符将操作数坐标映射到 swizzle 后的共享内存原子：描述符的步长选择原子，swizzle 选择原子内部的字节位置](../img/smem_descriptor.svg)

## Hopper 输出仍然使用寄存器

Hopper 改变了输入路径，但累加器仍然在寄存器中。

`wgmma` 指令将累加器写入每线程寄存器片段。确切的片段大小和寄存器数量取决于指令形状，例如 `m64nNk16`，其中 N 改变累加器寄存器的数量。但基本思想与 Ampere 相同：尾声部分消费的是一个寄存器片段。

因此 Hopper 有一个混合的布局模型。输入操作数可以直接来自共享内存描述符，swizzle 由硬件描述。输出累加器仍然是一个寄存器布局问题。

Blackwell 改变了输出侧。

## Blackwell：`tcgen05` 和 TMEM

Blackwell 保留了数据操作数的共享内存描述符思想。A 和 B 仍然以 Tensor Core 所期望的布局在共享内存中准备。某些模式也可以从 TMEM 读取 A 操作数。

主要的变化在于累加器。`tcgen05.mma` 将其累加器写入张量内存（Tensor Memory，简称 TMEM），而不是将其保持为长期存在的寄存器片段。在计算阶段，累加器停留在 TMEM 中。尾声部分稍后使用 `tcgen05.ld` 将其加载回寄存器。

这将对输出布局的问题从寄存器转移到了 TMEM。内核必须分配 TMEM，选择合适的 TMEM 布局，等待 MMA 完成，然后使用匹配的 `tcgen05.ld` 路径恢复累加器片段供尾声部分使用。

关于 `cta_group::1` 和 `cta_group::2` 如何将累加器划分给一个或两个 CTA 的细节在 {ref}`chap_tensor_cores` 中介绍。与早期世代差异最大的布局是块缩放的缩放因子布局。

## TMEM 中的 Scale Factor 布局

块缩放 MMA 模式，例如 `mxfp8` 和 `nvfp4`，增加了缩放因子操作数。除了 A 和 B，MMA 还读取：

```text
SFA(M, SFK)
SFB(N, SFK)
```

其中 `SFK` 是 K 方向缩放块的数量。

数据操作数 A 和 B 位于共享内存中。缩放因子位于 TMEM 中。这使得它们有一个不同的移动路径。

TMA 从全局内存加载到共享内存。它不直接加载到 TMEM。因此缩放因子通常分两步移动：

```text
全局内存到共享内存，使用 TMA
共享内存到 TMEM，使用 tcgen05.cp
```

只有在此拷贝之后，缩放因子才到达 `tcgen05.mma` 期望读取的内存空间。

TMEM 缩放因子布局使用 TMEM 硬件坐标 Lane 和 Col。在 TIRx 布局记号中，这些轴写作 `TLane` 和 `TCol`。

一个 128 行的缩放向量被压缩到一个 32 线程的组中，然后复制到 TMEM 的四个 32 线程窗口。在布局记号中，核心模式为：

```text
S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
```

分片将基本的 32 行组置于：

```text
TLane = r
TCol  = s
```

复制项在 Lane 偏移量 0、32、64 和 96 处添加副本：

```text
TLane = r + 32 * q, 其中 q 属于 {0, 1, 2, 3}
TCol  = s
```

这就是 `warpx4` 广播模式。相同的紧凑缩放因子组在整个 128 线程 TMEM 空间中变为可见。

在 32 位 `TCol` 单元内部还有字节打包。打包取决于 `scale_vec` 模式：

```text
1X：一个缩放值广播到整个 32 位单元
2X：两个缩放值被打包，每个重复一份
4X：四个 K-block 缩放值被打包
```

![scale_vec 字节打包：1X 将一个缩放广播到 4 字节单元；2X 打包两个缩放，每个重复；4X 打包四个 K-block 缩放](../img/sf_scale_vec.svg)

这种打包在 Ampere 或 Hopper 上没有直接的对应物，因为那些世代没有用于 `tcgen05` 块缩放 MMA 的 TMEM 缩放因子操作数。

在 `cta_group::2` 中，缩放因子跟随它们所缩放的数据。SFA 缩放 A，因此它按 M 划分到两个 CTA 上，与每个 CTA 拥有的 A 行相匹配。SFB 缩放 B，而 B 由计算的两个 CTA 半部分共享，因此 SFB 被多播到两个 CTA（{ref}`chap_tensor_cores`）。

## 一个反复出现的 Fragment

尽管周围的内存路径发生了变化，有一种结构一再出现：m8n8 风格的寄存器片段。

在 Ampere 上，`ldmatrix` 构建了该片段，供 `mma.sync` 读取。

在 Hopper 上，`wgmma` 将其累加器作为寄存器片段写入供尾声部分使用。

在 Blackwell 上，累加器在计算期间位于 TMEM 中，但 `tcgen05.ld` 在尾声部分处理和存储之前将其加载回寄存器片段（{ref}`chap_tmem`）。

因此该片段并没有消失。它的角色发生了变化。早期世代在整个计算阶段将累加器保留在那里。Blackwell 主要将其用于 TMEM 和尾声部分的边界处。

## 主线

在 Ampere 上，内核显式地构建 Tensor Core 寄存器片段。共享内存 swizzle 主要通过内核自身的索引计算来实现。

在 Hopper 上，Tensor Core 可以通过矩阵描述符直接从共享内存读取操作数。Swizzle 变成了一种由 TMA 和 `wgmma` 共享的命名描述符格式。

在 Blackwell 上，输入侧仍然使用共享内存操作数，但累加器移到了 TMEM。块缩放 MMA 还增加了必须暂存到 TMEM 中的缩放因子操作数。

描述符并没有消除布局工作。它们使约定变得显式。内核仍然必须确保数据移动路径、内存布局和 Tensor Core 指令三者一致。写入 swizzled SMEM 数据块的 TMA 描述符、读取该数据块的 MMA 描述符，以及附加到缓冲区的布局，都必须描述相同的物理排列。

如果其中任何一项不一致，硬件仍然会运行。它只是会读取错误的字节，或者读取得很慢。这就是为什么布局不是 Tensor Core 内核的装饰品，而是指令接口的一部分。
