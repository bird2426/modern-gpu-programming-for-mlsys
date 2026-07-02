(chap_tirx_layout_api)=
# TIRx 布局 API

:::{admonition} 概述
:class: overview

- TIRx 布局 API 将 {ref}`chap_data_layout` 中的布局表示法转换为编译器对象。主要对象有 `TileLayout`、`SwizzleLayout` 和 `ComposeLayout`。
- `TileLayout` 描述了在命名硬件轴上的仿射放置。它由分片规格 `S[...]`、副本规格 `R[...]` 和可选的偏移量构成。
- 一个布局将一个逻辑坐标映射到一个或多个物理坐标。`layout.apply()` 对该映射求值。
- `SwizzleLayout` 描述了用于避免 bank 冲突的基于 XOR 的共享内存交错。`ComposeLayout` 将交错叠加在瓦片布局之上。
- 现成的构造函数如 `tmem_datapath_layout`、`tcgen05_atom_layout` 和 `wg_local_layout` 涵盖了内核中反复出现的硬件布局。
:::

{ref}`chap_data_layout` 介绍了本书中使用的表示法：一个瓦片形状、一组在命名轴上的步幅，以及一个可选的副本项，用于表示被复制而非分区的值。本章将该表示法转换为编译器使用的 API。

目标是书页上的表示法和内核中的代码看起来几乎一致。当你编写这样一个布局时：

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

你不仅仅是在写一段说明。你正在构造一个可以附加到缓冲区的 `TileLayout` 对象。之后，每个触及该缓冲区的瓦片操作都可以从布局中读取其放置信息。放置信息只需编写一次、检查一次，并由编译器重复使用。

布局可以在从池中分配时附加，也可以在声明缓冲区时附加：

```python
pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)
```

从那时起，缓冲区就携带了它的物理放置信息。瓦片操作无需重复说明每个元素的位置。

布局对象位于同一个模块中：

```python
from tvm.tirx.layout import (
    TileLayout,
    SwizzleLayout,
    ComposeLayout,
    S,
    R,
    laneid,
    warpid,
    tid_in_wg,
    TLane,
    TCol,
    m,
    tcgen05_atom_layout,
    tmem_datapath_layout,
)
```

API 背后有一个核心理念。布局不必将逻辑索引映射到单个物理地址。它将逻辑索引映射到一组在命名轴上的物理坐标。在通常情况下，该集合只有一个元素。当存在复制时，同一个逻辑元素会有多个物理放置位置。

这就是布局模型包含三个部分的原因：分片、副本和偏移量。分片放置元素。副本将其复制到额外的坐标上。偏移量将整个放置位置进行平移。

## 布局示例

下面的例子展示了 API 的基本形态。

TMEM 中的累加器可以写成在 TMEM 轴上的直接放置：

```python
acc = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])
```

这里，逻辑行映射到 `TLane`，逻辑列映射到 `TCol`。在 {ref}`chap_tmem` 中，硬件坐标被称为 Lane 和 Col。在 TIRx 布局表示法中，这些硬件轴写作 `TLane` 和 `TCol`。

一个块缩放 MMA 缩放因子布局使用复制：

```python
scale_factor_layout = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
)
```

分片将一个 32 行的组放置在 TMEM 中。副本以 32 个 lane 为步幅将该组重复四次，因此这个 32 行的组在完整的 128 个 lane 的 TMEM 空间中都是可见的。

张量核心寄存器片段可以分布在 lane 和 warp 上：

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

同一个物理轴可以出现多次。在这个例子中，两个不同的迭代都对 `laneid` 有贡献。没有显式轴的步幅使用默认内存轴 `m`。

在实际内核中，常见的硬件布局通常来自构造函数：

```python
acc = tmem_datapath_layout("D", 128, 256)

ld = tcgen05_atom_layout("32x32b", (128, 64), "float32")
```

这些构造函数返回普通的 `TileLayout` 对象。它们是便利设施，而非独立的机制。你可以检查返回的布局、将其与其他布局组合，或者在形状不常见时手动编写底层的 `S[...]` 和 `R[...]` 形式。

## 交互式演示

在讲解机制之前，先有一个具体可以操作的东西会很有帮助。下面的演示让你选择一个预设布局、编辑逻辑形状和 `S` 或 `R` 项、选择数据类型和交错模式，并点击某个元素来查看它归属于哪个或哪些物理坐标。

```{raw} html
<p>
  <a class="reference external" href="../_static/tirx-layout-demo/index.html"
     target="_blank" rel="noopener"
     style="display:inline-block; padding:10px 18px; background:#3b82f6;
     color:#fff !important; font-weight:700; border-radius:8px;
     text-decoration:none;">▶ 全屏打开演示 ↗</a>
</p>
<iframe id="tirx-layout-demo-frame" src="../_static/tirx-layout-demo/index.html?notitle"
        style="width:100%; height:1040px; border:1px solid #dfe1e6;
        border-radius:10px; margin:10px 0 6px; display:block;"
        title="TIRx 交互式布局演示" loading="lazy"></iframe>
<script>
// The demo (viz-base.js) posts its content height; size the iframe to fit so
// there is no inner scrollbar. This demo is responsive (fills the width), so
// only the height follows content.
(function () {
  var f = document.getElementById('tirx-layout-demo-frame');
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    if (f && e.source === f.contentWindow) f.style.height = d.height + 'px';
  });
})();
</script>
```

这个演示很有用，因为 API 的大部分内容只是演示内容的一个精确版本。一个逻辑元素进入布局。布局将其展平、在其迭代之间拆分、在命名轴上累加坐标，然后根据需要进行复制。

## TileLayout

`TileLayout` 是主要的仿射布局对象。它通常使用与文本中相同的表示法来编写：

```python
TileLayout(S[shape : strides])
```

`S` 项是分片规格。你可以这样理解：获取一个此形状的逻辑瓦片，并使用这些步幅将其放置在命名轴上。

当一个值需要出现在多个位置时，分片规格会扩展一个副本规格：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])
```

还可以添加一个可选的偏移量：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride] + offset)
```

在底层，这些部分由迭代（iter）表示。一个迭代是一个三元组：

```text
(extent, stride, axis)
```

它描述了在一个命名轴上的带步幅遍历。extent 表示迭代有多少个位置。stride 表示每一步移动多远。axis 表示正在改变哪个硬件坐标。

一个布局有三个部分。

### 分片 (Shard)

分片，即 `D`，是由 `S[...]` 构建的部分。它将逻辑索引分布在一个或多个迭代上，并产生基础的物理坐标。

例如：

```python
S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
```

有四个分片迭代。它们的 extent 分别是 `8`、`2`、`4` 和 `2`。它们的步幅将数据分别放置在 `laneid`、`warpid`、`laneid`（再次）和默认内存轴 `m` 上。

这是对普通形状-步幅规则的推广。区别在于，步幅是附加到命名硬件轴上，而不是附加到单个扁平地址上。

### 副本 (Replica)

副本，即 `R`，描述了同一逻辑元素的额外物理拷贝。副本迭代与逻辑索引无关。它们列举硬件空间中的额外偏移量。

例如：

```python
R[2 : 4@warpid]
```

在 `warpid` 轴上创建了两个相隔四个 warp 的拷贝。

复制并非为了方便而耍的花招。它描述了真实的硬件行为。某些数据会在 warp、lane 或内存区域之间广播。逻辑到物理的映射天然支持这一点，因为一个逻辑元素可以映射到一组物理坐标。

### 偏移量 (Offset)

偏移量，即 `O`，是一个加到每个结果上的固定坐标。

例如：

```python
5@warpid
```

将整个放置位置在 `warpid` 轴上平移五个单位。

偏移量用于将瓦片放置在选定的基准坐标上、为独占使用预留区域、或者描述在同一资源中紧邻另一个瓦片开始的瓦片。

### 组合各部分

布局按顺序应用这三个部分。

首先，分片计算基准坐标。然后，副本将该坐标扩展为零个或多个额外拷贝。最后，偏移量平移每个坐标。

对于逻辑坐标 `x`，结果是：

```text
L(x) = { D(x) + r + O | r in R }
```

如果没有副本，`R` 只包含零偏移量，因此结果是一个单元素集合。如果有副本，结果中每个副本位置对应一个坐标。

在 TIRx 语法中，一个完整的布局可以这样写：

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

从左到右阅读，分片放置逻辑瓦片，副本在四个 warp ID 之外创建一个第二份拷贝，偏移量将整个放置位置平移到从 `warpid = 5` 开始。

如果迭代已经构建为对象，则可以直接构造相同的布局：

```python
TileLayout.from_iters(shard, replica, offset)
```

大多数用户代码使用 `S[...]` 和 `R[...]` 表示法，因为它更接近数学形式。

## 命名轴

布局中的轴不是匿名维度。每个轴都命名了一个真实的硬件坐标或编译器级的放置坐标。

例子包括：

```text
bx, by, bz
cbx, cby, cbz
tx
warpid
laneid
wgid
tid_in_wg
wid_in_wg
m
P, F
Bank
TLane, TCol
```

网格轴如 `bx`、`by` 和 `bz` 将工作分布到 CTA 上。集群轴如 `cbx`、`cby` 和 `cbz` 在 CTA 集群内分配工作。线程轴如 `tx`、`warpid`、`laneid`、`tid_in_wg` 和 `wid_in_wg` 描述了 CTA 或 warp 组内部的归属关系。轴 `m` 是默认的线性内存轴。`P` 和 `F` 用于二维暂存器风格的放置。`Bank` 命名共享内存 bank。`TLane` 和 `TCol` 是 TMEM Lane 和 Col 坐标的 TIRx 布局名称。

轴名称是布局的一部分。这一点很重要，因为两个具有相同整数值的坐标可能代表不同的硬件含义。`1@tx` 与 `1@tid_in_wg` 不同。`1@laneid` 与 `1@TLane` 不同。布局明确地维护了这些含义。

## 正向映射

评估一个布局意味着接收一个逻辑坐标并计算它在物理上落在哪里。API 方法是：

```python
layout.apply(*coord)
```

对于没有复制的布局，结果是一个坐标字典。有复制时，结果是一组坐标字典。坐标字典将轴名称映射到整数位置，例如：

```python
{"laneid": 7, "warpid": 2, "m": 1}
```

求值规则包含四个步骤。

第一步，按行主序展平逻辑坐标。对于逻辑坐标：

```text
x = (x0, x1, ..., xr-1)
```

在逻辑形状内：

```text
(S0, S1, ..., Sr-1)
```

扁平索引为：

```text
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1
```

第二步，将该扁平索引在分片 extent 上进行拆分。如果分片 extent 为：

```text
(e0, e1, ..., en-1)
```

则拆分产生分量：

```text
c0, c1, ..., cn-1
```

使用相同的行主序顺序在分片 extent 上展开。

第三步，将每个分量使用其步幅累加到其轴上。如果分片迭代 `k` 的 extent 为 `ek`、步幅为 `sk`、轴为 `ak`，则分量 `ck` 贡献：

```text
ck * sk @ ak
```

所有对同一轴的贡献相加。然后加上偏移量。

第四步，应用副本迭代。每个副本迭代贡献一个与逻辑坐标无关的额外偏移量。如果有多个副本迭代，布局会枚举所有组合。

这个规则的一个有用推论是，布局不需要硬编码输入形状。它只需要逻辑瓦片的总元素数与分片 extent 的乘积相同即可。一旦满足该条件，展平和拆分就定义了映射关系。

## 案例研究：张量核心寄存器瓦片

考虑一个逻辑 `(8, 16)` 瓦片分布在两个 warp 上，每个 warp 有 32 个 lane。每个 lane 拥有一个小的寄存器片段。寄存器槽位由默认内存轴 `m` 表示。

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

从 `(8, 16)` 瓦片中取一个逻辑元素 `(i, j)`。

行主序扁平索引为：

```text
flat = 16 * i + j
```

按分片 extent `(8, 2, 4, 2)` 拆分得到：

```text
c0 = i
c1 = floor(j / 8)
c2 = floor(j / 2) mod 4
c3 = j mod 2
```

分片贡献为：

```text
laneid = 4 * c0 + c2
warpid = c1
m      = c3
```

加上偏移量 `5@warpid` 后，变为：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5
m      = j mod 2
```

副本项：

```python
R[2 : 4@warpid]
```

给 `warpid` 加上 `0` 或 `4`。因此完整的映射为：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5 + 4 * r, 其中 r in {0, 1}
m      = j mod 2
```

分片将瓦片放置在 warp 5 和 6 上。然后副本将其复制到 warp 9 和 10。因此同一逻辑元素出现在两个 warp 位置上。

这个例子说明了为什么该模型使用一组物理坐标。复制不能自然地用从物理坐标到逻辑坐标的函数来表示。它天然地表现为从一个逻辑坐标到多个物理坐标的函数。

## 案例研究：Blackwell 张量内存

相同的布局模型适用于内存放置。轴不一定是线程轴，也可以是内存轴。

TMEM 通过硬件 Lane 和 Col 坐标寻址。在 TIRx 布局表示法中，这些轴写作 `TLane` 和 `TCol`。

考虑这个布局：

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)
```

如果逻辑瓦片形状为 `(2, 128, 112)`，拆分分量就是逻辑坐标本身。对于元素 `(a, l, c)`，映射为：

```text
TLane = l
TCol  = 112 * a + c
```

extent 为 128、步幅为 `1@TLane` 的迭代填充了 128 个 TMEM Lane 行。extent 为 2、步幅为 `112@TCol` 的迭代和 extent 为 112、步幅为 `1@TCol` 的迭代共同覆盖了 224 列：

```text
TCol in [0, 224)
```

224 列的跨度是有意设计的。TMEM 布局不必是 2 的幂次。块缩放 FP8 GEMM 可能会选择 224 列的累加器，因为完整的 256 列瓦片无法为两个累加器阶段加上缩放因子留出足够的 TMEM 容量。布局 API 可以直接表达这种形状。

## 缩放因子布局

上面的累加器布局是一个纯粹的放置。每个逻辑累加器元素映射到一个 TMEM 坐标。块缩放 MMA 的缩放因子则不同，因为同一物理组可能需要在多个 warp 窗口上可见。这就是复制发挥作用的地方。

一个紧凑的缩放因子布局可以写成：

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)
```

分片将一个 32 行的缩放因子组放置在 TMEM 中：

```text
TLane = r
TCol  = s
```

对应一个逻辑缩放坐标 `(r, s)`。

副本项创建了四个相隔 32 个 lane 的拷贝：

```text
TLane = r + 32 * q, 其中 q in {0, 1, 2, 3}
TCol  = s
```

因此这个 32 行的组在 TMEM lane 0 到 31、32 到 63、64 到 95 以及 96 到 127 上都是可见的。这就是 `warpx4` 广播模式（{ref}`chap_layout_generations`）。四个 warp 大小的 TMEM lane 窗口各自看到同一个缩放因子组。

在完整的块缩放 MMA 布局中，这个原子与沿 M 行和 K 缩放因子组的外层迭代组合在一起。根据缩放因子数据类型的不同，多个缩放因子可能被打包到一个 32 位 `TCol` 单元格中。例如，fp8 缩放因子可以将四个值打包到一个 32 位列单元格中。可选的步幅为零的复用和管道深度迭代可以进一步描述缩放因子在多个 MMA 和双缓冲之间的复用情况。

重要的是，同一个 `TileLayout` 模型描述了这两种情况。累加器是 TMEM 中的一个单一放置。缩放因子是同一 TMEM 地址空间中的复制放置。

## 现成布局

大多数内核不会手写每一个硬件布局。TIRx 为经常出现的布局提供了构造函数。

```python
tmem_datapath_layout(datapath, rows, cols)
```

返回由 `tcgen05.mma` 编写的 TMEM 累加器布局。`datapath` 参数选择行放置模式。例如，`"D"` 对应 `M = 128` 的单位风格放置，而 `"F"` 对应 `M = 64` 的分散式放置。

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
```

返回由 `tcgen05.ld` 或 `tcgen05.st` 原子移动的寄存器瓦片布局。指令形状的示例包括 `.32x32b`、`.16x64b`、`.16x128b` 及相关形式。在 DSL 层面，这是一个 warp 组分布的瓦片。在降级过程中，它变为四条 warp 集体型的 `tcgen05.ld` 或 `tcgen05.st` 指令，每个 warp 一条，每条处理其各自的 32 个 TMEM lane。

```python
wg_local_layout(cols, rows=128)
```

返回一个 warp 组本地的寄存器瓦片，通常在 `tid_in_wg` 上每线程一行。

这些辅助函数的存在是为了避免手动重复编写常见的硬件映射。它们并不隐藏模型。每个辅助函数都返回一个普通的 `TileLayout`，由上述相同的 `S` 和 `R` 片段构建而成。

## SwizzleLayout 和 ComposeLayout

`TileLayout` 是仿射的。它可以表达在命名轴上的步幅、复制和偏移量。这对于许多放置已经足够，包括线程片段、TMEM 瓦片和紧凑的缩放因子布局。

共享内存交错需要别的东西。用于避免 bank 冲突的交错不是仿射步幅模式。它是一种基于 XOR 的线性共享内存地址置换。

因此 TIRx 将交错作为一个独立的布局对象：

```python
SwizzleLayout(...)
```

并将其与瓦片布局组合：

```python
ComposeLayout(swizzle, tile)
```

瓦片布局首先产生一个线性内存地址。然后交错对该地址进行置换。保持这两层分离，比将 XOR 置换强行塞入仿射布局模型要更清晰。

## 为什么需要交错

共享内存被分为 32 个 bank，每个 bank 字为 4 字节。当一个访问请求中不同 lane 触及同一 bank 中的不同地址时，该访问会因 bank 冲突而被串行化。

一个普通的行主序瓦片可能在结构上造成这种冲突。考虑一个 `(8, 64)` float16 瓦片，行主序布局：

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])
```

逻辑元素 `(i, j)` 的线性元素地址为：

```text
m = 64 * i + j
```

每行有 64 个 float16 值，即 128 字节。这正好是一条完整的共享内存 bank 线。如果一个 warp 沿着一列读取，具有固定的 `j`，则每行步进一个完整的 128 字节线。bank 索引会重复，因此整列的读取会坍缩到跨行的同一个 bank 上。

交错通过使低位地址依赖于更高位的行值来改变这种情况。原本会反复落在同一 bank 上的列会被分散到不同的 bank。

## 交错变换

`SwizzleLayout` 由三个整数参数控制：

```text
per_element = M
swizzle_len = B
atom_len    = S

```

输入是一个线性元素地址 `m`。

`m` 的低 `M` 位保持不变。这保留了一个小的连续元素组。高位被下移到一个临时值：

```text
x = m >> M
```

然后将 `x` 的 `[S, S + B)` 位置的位组与 `x` 的 `[0, B)` 位置的位组进行 XOR。然后将未改变的低 `M` 位放回去，形成交错后的地址。

等价地：

```text
mask = (1 << B) - 1

low  = m & ((1 << M) - 1)
x    = m >> M
x2   = x ^ ((x >> S) & mask)

addr = (x2 << M) | low
```

为了使布局格式正确，`S` 必须至少为 `B`。

这个变换的目的不是改变瓦片中的逻辑元素。它改变的是这些元素在共享内存中的位置。MMA 仍然读取相同的逻辑瓦片。交错使得物理 bank 模式变得更好。

## 选择交错参数

在正常使用中，交错参数是根据数据类型和共享内存交错模式来选择的。常见模式有 32 字节、64 字节和 128 字节交错。

`per_element` 参数的选择使得一个小型向量大小的组保持连续。对于 float16，一个 16 字节的向量包含 8 个元素，因此：

```text
M = log2(8) = 3
```

使用 128 字节交错时，布局使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

这保持 16 字节的向量组不变，同时对更大的共享内存地址模式进行足够的置换，以打破列 bank 冲突。

大多数代码不应手动推导这些参数。数据类型和描述符模式通常决定了它们。对程序员来说重要的是，TIRx 布局中的交错、TMA 描述符和 MMA 期望三者需要一致。

因此，一个交错后的共享内存分配看起来像：

```python
tile = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

layout = ComposeLayout(swizzle, tile)
```

组合后的布局就是附加到共享内存缓冲区的对象。

## 元素的 Bank 和线

为了判断交错是否有帮助，将交错后的元素地址转换回共享内存 bank。

设 `addr` 为交错后的元素地址，`b` 为元素大小（字节）。字节地址为：

```text
byte = addr * b
```

bank 为：

```text
bank = floor(byte / 4) mod 32
```

128 字节的 bank 线为：

```text
line = floor(byte / 128)
```

对于 float16，`b = 2`，因此 bank 公式变为：

```text
bank = floor(addr / 2) mod 32
```

这就是下面工作示例中使用的公式。

## 工作示例：`(8, 64)` float16 瓦片上的 128B 交错

回到行主序 float16 瓦片：

```text
m = 64 * i + j
```

使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

变换变为：

```text
x    = m >> 3
addr = ((x ^ ((x >> 3) & 7)) << 3) | (m & 7)
```

由于：

```text
m = 64 * i + j
```

我们可以写出：

```text
q = floor(j / 8)
r = j mod 8
```

交错后的地址为：

```text
addr = 64 * i + 8 * (q xor i) + r
```

现在看列 `j = 0`。那么 `q = 0` 且 `r = 0`，因此：

```text
addr = 72 * i
```

对于 float16，bank 为：

```text
bank = floor(addr / 2) mod 32
```

因此八行映射到：

```text
i = 0: bank 0
i = 1: bank 4
i = 2: bank 8
i = 3: bank 12
i = 4: bank 16
i = 5: bank 20
i = 6: bank 24
i = 7: bank 28
```

这一列现在触及八个不同的 bank。冲突消失了。

如果没有交错，同一列的地址为：

```text
m = 64 * i
```

因此：

```text
bank = floor(64 * i / 2) mod 32 = 0
```

每行都落在 bank 0 上，因此访问被串行化。交错只改变了物理放置位置，但这足以将列访问变成无冲突访问。

这种保证依赖于按照设计方式使用交错。数据类型、交错宽度和访问形状必须与 TMA 和 MMA 描述符模式匹配。128 字节 float16 交错是围绕相关的 16 字节行块和张量核心访问模式设计的。它并不能保证任意共享内存访问都变得无冲突。本章顶部的演示可以直观地看到这一点：选择一种数据类型和交错模式，观察一列在没有交错时坍缩到一个 bank 上，然后在应用匹配的交错后分散到整个 bank 视图。

## 设计原理

布局 API 遵循三个设计选择。

第一，它支持通用形状。硬件瓦片并不总是 2 的幂次。全局张量、共享内存阶段、TMEM 累加器和缩放因子缓冲区通常具有由容量限制或算法选择决定的形状。布局模型将这些形状视为正常情况。

第二，映射从逻辑坐标到物理坐标。这个方向很重要，因为复制很常见。一个逻辑元素可能存在于多个物理位置。逻辑到物理的映射直接将其表示为一组坐标。

第三，硬件轴是显式的。布局不使用匿名维度并依赖上下文事后解释它们。`tx`、`tid_in_wg`、`laneid`、`warpid`、`TLane` 和 `TCol` 之间的区别被直接写入布局本身。

合法性和可行性检查并非布局对象单独的责任。布局可以说明数据放在哪里。更高级的瓦片原语决定给定的操作是否能够合法且高效地使用该放置。这种分离使布局 API 保持小巧，同时仍然为编译器提供足够的信息来分派真实的硬件操作。
