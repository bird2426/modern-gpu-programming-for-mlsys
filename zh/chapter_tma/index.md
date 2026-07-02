(chap_tma)=
# 异步数据搬运：TMA


:::{admonition} 概览
:class: overview

- TMA 是用于在全局内存和共享内存之间进行异步 tile 拷贝的硬件引擎。一个线程发起拷贝，引擎负责搬运字节。
- TMA 拷贝由张量映射描述符描述。该描述符告诉引擎全局张量的形状、步幅、tile 坐标以及共享内存 swizzle 模式。
- 在加载路径上，TMA 可以在写入共享内存时对 tile 进行 swizzle，从而使 tile 以 Tensor Core 期望的布局存放。
- TMA 加载通过带有字节计数跟踪的 `mbarrier` 完成。TMA 存储使用提交组和等待组。
:::

Tensor Core 只有在准备好数据供其消费时才有用。在 GEMM 或注意力内核中，一旦流水线填满，计算可能是计算密集型的（{ref}`chap_performance`），但流水线只有在下一个操作数 tile 及时到达时才能保持填满。

移动 tile 的旧方法是让线程自行拷贝。每个线程计算地址、从全局内存发起加载，并将值存储到共享内存中。这种方法可行，但它将 warp 指令消耗在地址算术和拷贝簿记上，而不是计算上。它还使拷贝路径暴露在应该为 Tensor Core 提供数据的同一个 warp 的指令流中。

张量内存加速器，简称 TMA，将这项工作移入硬件拷贝引擎。一个线程发起 tile 拷贝。然后拷贝引擎在全局内存和共享内存之间异步移动一个矩形 tile。在引擎搬运字节的同时，CTA 的其余部分可以继续执行其他工作。

TMA 还处理了部分布局问题。Tensor Core 不仅需要在共享内存中有正确的值，还需要它们以正确的共享内存布局存放。在加载路径上，TMA 可以在写入 tile 时应用共享内存 swizzle。这使 tile 能够直接以后续 MMA 所期望的布局存放。

```{raw} html
<iframe src="../demo_zh/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*交互式：TMA 将 tile 从全局内存拷贝到共享内存。切换 swizzle 模式，悬停源单元格查看其在共享内存中的位置。*

## 一个线程发起，硬件搬运 Tile

TMA 拷贝始于一个发起线程。该线程不会遍历 tile 中的所有元素。它向硬件提供拷贝的描述，然后 TMA 引擎执行传输。

主要输入是张量映射描述符。该描述符描述了全局张量以及如何从中读取 tile。它记录了诸如张量形状、步幅、元素大小、tile 形状和 swizzle 模式等信息。发起线程还提供了 tile 应存放的共享内存地址。

指令发起后，拷贝异步运行。发起线程可以继续执行。CTA 中的其他线程也可以继续。传输现在由 TMA 引擎负责，而不是由普通的加载和存储指令循环来处理。

这为内核提供了两种不同的方式来表达相同的逻辑操作"拷贝此 tile"。

一种路径是线程拷贝。线程协作从全局内存加载并存入共享内存。这使内核能够直接控制每次访问，但会消耗线程指令和寄存器用于地址计算。

另一种路径是 TMA 拷贝。一个线程发起传输，硬件拷贝引擎执行矩形拷贝。这是大型规则 tile 的自然路径，特别是 Tensor Core 内核使用的操作数 tile。

这两种路径具有不同的同步规则和不同的性能行为。在它们之间选择是一个调度决策。布局告诉内核它需要什么样的内存排列方式。作用域告诉它哪些线程或 CTA 参与。调度决定拷贝是由普通线程代码实现还是由 TMA 实现。

## Swizzled Layout

移动 tile 是不够的。tile 还必须以 Tensor Core 能够高效读取的布局放置在共享内存中。

这就是 TMA swizzle 的用武之地。当 TMA 将 tile 写入共享内存时，它可以置换共享内存地址模式。全局内存 tile 仍然是一个逻辑矩形，但共享内存中的目标布局可以被 swizzle。

Swizzle 模式是 TMA 描述符的一部分。描述符设置好后，发起线程无需手动应用 swizzle。引擎会在字节写入共享内存时自动应用。

重要的要求是一致性。TMA 描述符、共享内存 tile 布局以及后续的 MMA 指令都必须描述相同的布局（{ref}`chap_data_layout`）。如果 TMA 用一种 swizzle 写入 tile，但 MMA 以另一种方式读取它，硬件仍然会精确地执行被要求的操作，只是字节的排列方式不适合计算。

这正是布局符号不仅仅是簿记工具的地方。DSL 使用的布局必须与 TMA 描述符和 Tensor Core 指令使用的硬件布局一致。例如，如果内核说操作数 tile 以 128 字节 swizzle 布局存储，那么 TMA 描述符必须使用匹配的 swizzle 模式，并且 MMA 调度必须期望相同的共享内存排列方式。上方的演示让你可以在无 swizzle 和 128 字节 swizzle 之间切换；悬停源元素可以查看应用 swizzle 后的存放位置。

理解 swizzle 的一种有用方式是：TMA 并没有改变逻辑 tile。它改变的是逻辑元素在共享内存中的物理存放位置。后续的 MMA 仍然消费相同的逻辑 A 或 B tile。Swizzle 仅决定该 tile 如何在共享内存 bank 之间排列。

## 用 3D TMA 表达 Tiling 和 Swizzling

普通的 TMA 拷贝移动的是平坦的 2D tile，但 Tensor Core 期望的共享内存布局通常是分 tile 为 swizzle 原子的（来自 {ref}`chap_data_layout` 的 8 个 128 字节原子）。TMA 通过一个额外的描述符维度来处理这个问题。**3D TMA** 将共享内存盒子描述为 `(group, row, col)`，其中 group 维度横跨原子，内层两个维度在一个原子内寻址。单个 3D 拷贝既可以将 tile 按原子逐个布局（tiling），又可以在每个原子内应用 swizzle，从而使数据以 MMA 期望的布局到达，无需单独的 tiling 或 swizzle 步骤。

```{raw} html
<iframe class="demo-tma3d" src="../demo_zh/tma_3d.html" title="Tiling and swizzling with 3D TMA" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*交互式：3D TMA 拷贝，以 (group, row, col) 寻址，分 tile 到 swizzled 共享内存中。*

选择 swizzle **格式**与此 tiling 相关。更宽的 swizzle 将一列分散到更多 bank 上，因此在适用时 128 字节 swizzle 是默认选择，但 N 字节原子需要 tile 的连续维度来填充它。因此，由于形状约束而较小的 tile 无法使用 128 字节 swizzle，必须降级到 64 字节或 32 字节：经验法则是选择 tile 能填充的最大 swizzle（{ref}`chap_data_layout`）。下面的演示直接展示了这个约束：在 16 x 16 tile 上使用 128 字节 swizzle，只有当 tile 被拆分为匹配原子的 16 x 8 组时才能变为无冲突。

```{raw} html
<iframe class="demo-tma3d" src="../demo_zh/tiling_constraint.html" title="Swizzle imposes a tiling constraint" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*交互式：16 x 16 tile 上的 128 字节 swizzle，分 tile 为 16 x 8 组后变为无冲突。*

## 完成通知：Load

拷贝是异步的，因此仅发起是不够的。消费者不能仅仅因为 TMA 指令已发起就读取共享内存中的 tile。只有在引擎完成写入字节后，tile 才是安全的可读状态。

对于 TMA 加载，完成信号是 `mbarrier`（{ref}`chap_async_barriers`）。

通常的流程是：

1. 为流水线阶段初始化或重用 `mbarrier`；
2. 告诉 barrier TMA 传输预计写入的字节数；
3. 发起 TMA 加载；
4. 让 TMA 引擎在字节到达时更新 barrier；
5. 让消费者在读取共享内存 tile 之前等待 barrier 阶段。

字节计数通过如下操作设置：

```text
mbarrier.arrive.expect_tx(bytes)
```

这完成两个任务。它记录了预期的传输大小，同时也执行了发起线程在 barrier 上的到达。barrier 不仅因为此调用发生而完成。它仍等待 TMA 引擎报告预期的字节已到达。

随着传输的进行，引擎对 barrier 执行 complete-tx 更新。barrier 阶段只有在两个条件都满足时才会翻转：到达计数已满足，且待处理字节计数达到零。

消费者然后在该 barrier 上等待。一旦针对预期阶段的等待完成，共享内存 tile 就就绪了。此时 MMA 路径可以安全地读取它。

![TMA 加载同步流程](../../img/tma_sync_flow.png)

这是与其他异步生产者-消费者交接相同的 barrier 模型。生产者是 TMA 引擎。消费者是 MMA 路径或任何其他读取共享内存 tile 的代码。Barrier 是它们之间的显式交接点。

## 完成通知：Store

TMA 存储将数据向相反方向移动，从共享内存到全局内存。它们也是异步的，但完成机制不同。

TMA 加载通常服务于同一内核内的消费者。MMA 路径需要知道共享内存 tile 何时就绪。这就是加载路径使用 `mbarrier` 的原因。

TMA 存储通常将最终数据写出到全局内存。通常没有立即的内核内消费者在等待存储结果。内核主要需要知道的是何时可以安全地重用共享内存缓冲区或完成存储序列。

为此，TMA 存储使用提交组和等待组。内核发起一个或多个存储，提交组，然后等待组排空。等待完成后，从内核的角度来看，该组中的存储已完成，并且存储使用的共享内存区域可以安全地重用。

所以规则很简单：

```text
TMA 加载：通过带有字节计数跟踪的 mbarrier 等待
TMA 存储：通过提交组和等待组等待
```

两种机制在不同的交接点服务于相同的目的。加载需要使共享内存 tile 对后续消费者可见。存储需要确保在内核重用源存储或依赖存储已排空之前，传出传输已完成。

## 为什么 TMA 对 Pipelining 很重要

TMA 作为流水线的一部分时最为有用。内核可以在 Tensor Core 计算当前 tile 的同时为未来的 tile 发起加载。加载在后台运行。计算在前台运行。barrier 在未来的 tile 成为当前 tile 时将两者连接起来。

典型的 GEMM 循环重复使用这种结构。共享内存的一个阶段保存 MMA 当前消费的 tile。另一个阶段正在被 TMA 填充。随着循环推进，角色轮换。在 MMA 读取一个阶段之前，它等待该阶段的加载 barrier。在 TMA 覆盖一个阶段之前，内核确保之前的消费者已完成对该阶段的使用。

这就是 TMA 和 `mbarrier` 在 Blackwell 和 Hopper 风格的内核中通常一起出现的原因。TMA 为内核提供了异步拷贝引擎。barrier 为内核提供了精确知道拷贝字节何时就绪的方式。
