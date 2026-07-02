(chap_async_barriers)=
# 异步协作：mbarrier

<!--
翻译模板

英文源文件：chapter_async_barriers/index.md
建议：保留 mbarrier API、phase、arrival count 和 demo iframe。
-->

:::{admonition} 概览
:class: overview

- TMA 和 Tensor Core 是异步的，因此发起工作不等于完成工作，消费者需要一个显式的完成信号。
- mbarrier 就是这样的信号：生产者到达，消费者等待，它跟踪到达计数和（对于 TMA 而言）字节计数。
- 每个 barrier 携带一个 *phase*，每轮翻转；等待正确的 phase 是安全地门控消费者的方式。
:::

TMA（{ref}`chap_tma`）和 Tensor Core（{ref}`chap_tensor_cores`）操作是异步的。当内核发起 TMA 加载或 `tcgen05` MMA 时，发起线程不会等待操作完成。指令仅仅被提交给硬件引擎；实际的数据移动或矩阵操作与程序的其余部分并行继续。

这很有用，因为它让内存移动和计算可以重叠。这也意味着程序顺序不足以证明数据已就绪。后面的指令可能在前面的异步操作完成之前就已经执行。如果 TMA 仍在写入共享内存 tile 时 MMA 就开始读取它，MMA 将读到不完整的数据。如果 epilogue 在 Tensor Core 完成写入累加器之前读取 TMEM，它将读到错误的值。如果内核等待了错误的条件，它可能永远无法推进。

因此，内核在每个异步交接点都需要一个显式的完成信号。`mbarrier` 就是这样的信号。生产者在工作完成时到达 barrier，消费者在使用产生的数据之前等待该 barrier。同一机制用于 TMA 到 MMA 的交接、MMA 到 epilogue 的交接，以及跨流水线阶段的缓冲区重用。

Barrier 不仅仅是一次性的标志。它携带一个 phase 位，每次 barrier 完成一轮到达时该 phase 位就会翻转。Phase 的作用是让同一个 barrier 可以在多次循环迭代中被重用，而不会将一次迭代的完成与另一次迭代的完成混淆。

## mbarrier

`mbarrier`，即内存屏障（memory barrier），是一个存储在共享内存中的硬件同步对象。从概念上讲，它包含两个状态：一个到达计数器和一个 phase 位。计数器告诉 barrier 在当前轮中还有多少到达未完成。Phase 位告诉内核 barrier 当前处于哪一轮。

```{raw} html
<iframe src="../demo_zh/mbarrier_mechanism.html" title="mbarrier data structure and APIs" loading="lazy"
        style="width:100%; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*交互式：mbarrier 状态视图，显示到达计数器、phase 位以及 init、arrive 和 wait 操作；点击字段可聚焦查看。*

Barrier 从初始化开始。在 `init` 期间，内核设置该 barrier 应期望的到达次数。Barrier 从 phase 0 开始，其计数器加载为该期望的到达计数。从那时起，barrier 等待所有需要的生产者或资源使用者报告它们已完成。

到达操作减少了 barrier 仍在等待的工作量。内核的不同部分可以以不同方式到达 barrier，这种区别很重要。

对于 TMA 加载，通常的到达路径是 tx-count 到达。诸如 `mbarrier.arrive.expect_tx(bytes)` 的操作完成两件事。首先，它算作发起线程在 barrier 上的到达。其次，它记录了 TMA 引擎预计传输的字节数。Barrier 不仅仅因为发起线程已到达就完成。它还等待 TMA 引擎在传输完成时排空字节计数。Phase 只有在两个条件都满足后才翻转：普通到达计数已达到零，且待处理 tx 字节计数已达到零。

这就是为什么 `expect_tx` 不应被理解为"多一个普通到达"。它为异步拷贝设置了一个字节预算。硬件随后通过 complete-tx 更新来说明实际的拷贝完成情况。Barrier 只有在到达和字节传输都完成后才完成。

对于 Tensor Core 工作，到达路径不同。`tcgen05` MMA 不会仅仅因为 MMA 被发起就自动推进 barrier。内核必须显式地将 barrier 到达附加到提交路径上，例如使用 `tcgen05.commit.mbarrier::arrive` 操作。当该提交组完成时，Tensor Core 端执行 barrier 到达。如果内核忘记该提交到达，等待 barrier 的消费者将永远等待。

普通线程也可以直接到达 barrier。这用于普通线程代码是生产者的情况，或者一组线程宣布它已完成使用某个资源的情况。例如，消费者完成读取共享内存缓冲区后，它可以到达一个 barrier，告诉生产者该缓冲区可以自由重用。

等待是同一协议的消费者端。消费者等待，直到 barrier 已完成当前迭代所期望的 phase。只有到那时，读取数据或重用该 barrier 所保护的资源才是安全的。

重要的是，异步硬件不仅领先于程序运行；它还通过 barrier 报告完成状态。TMA 可以发出共享内存 tile 已就绪的信号。Tensor Core 工作可以发出 TMEM 结果已就绪的信号。普通线程可以发出缓冲区不再使用的信号。Barrier 为所有这些情况提供了相同的生产者-消费者模式：生产者到达，消费者等待。

## Phase Tracking

Barrier 通常不是为单次使用分配的。流水线化的 K 循环可能执行数百次相同的交接，为每次迭代分配一个新的共享内存 barrier 是不现实的。相反，内核保留一小组固定的 barrier，并在循环推进时重用它们。

Phase 位使得这种重用是安全的。

```{raw} html
<iframe src="../demo_zh/phase_tracking.html" title="mbarrier phase tracking" loading="lazy"
        style="width:100%; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*交互式：跨多个流水线迭代重用的 barrier，显示每完成一轮后 phase 位翻转。*

每次 barrier 完成当前轮的所有到达时，它翻转 phase：phase 0 变为 phase 1，phase 1 变为 phase 0，依此类推。等待操作检查消费者期望的 phase。该期望 phase 由内核保存在寄存器中。在一个阶段成功完成一轮等待后，内核在下一次使用该 barrier 之前翻转其本地 phase 值。

这防止了内核将旧的完成误认为是新的完成。假设一个 barrier 用于一次 TMA 加载并且已经完成。如果下一个循环迭代在不跟踪 phase 的情况下重用同一个 barrier，消费者可能观察到之前的完成并错误地认为新加载已就绪。Phase 位将这两轮分开。迭代 0 等待一个 phase，迭代 1 等待相反的 phase，迭代 2 再次等待第一个 phase，如此往复。

在实际的流水线中，簿记通常是按阶段进行的。内核有固定数量的共享内存阶段，相应固定数量的 barrier，以及寄存器中的一小组 phase 值。随着循环推进，每个逻辑迭代映射到一个物理阶段，phase 值告诉等待操作它正在等待该物理 barrier 的哪一轮。

这就是为什么后续的 GEMM 代码不需要每个 K tile 一个 barrier（{ref}`chap_gemm_async`）。它需要每个可重用阶段一个 barrier，加上 phase 跟踪。阶段索引选择共享内存缓冲区和 barrier。Phase 值将该阶段当前的使用与前一次使用区分开。

**与你的 agent 一起尝试**：给它一个两阶段流水线，让它跟踪四次迭代。对于每次迭代，列出阶段索引、本地 phase 值、barrier 何时翻转，以及在阶段重用前未切换 phase 会导致什么问题。

## 同步规则

一旦 barrier 和 phase 机制清晰了，tensor-core 内核中的同步模式就相当机械了。每当一条路径产生数据或释放资源而另一条路径将消费它时，交接必须显式化。

有三种常见情况。

第一种情况是线程代码为异步引擎产生数据。如果线程写入共享内存，而后续的 TMA 存储或 MMA 指令读取该共享内存，内核必须在引擎读取之前使线程的写入可见。这需要适当的线程级同步或 fence。确切的指令取决于交接的作用域，但原因始终相同：引擎不能在生产线程完成写入之前观察到共享内存缓冲区。

第二种情况是 TMA 为 MMA 产生数据。TMA 加载异步填充共享内存 tile。MMA 路径不能仅仅因为 TMA 指令已发起就推断 tile 已就绪。TMA 操作必须与 `mbarrier` 关联，并且 MMA 路径必须在读取 tile 之前等待该 barrier。

第三种情况是 MMA 为 epilogue 产生数据。`tcgen05` MMA 异步将其结果写入 TMEM。Epilogue 在 Tensor Core 完成相关工作之前不能安全地读取累加器。因此，MMA 提交路径在一个完成 barrier 上到达，epilogue 在读取 TMEM 之前等待该 barrier。

```{raw} html
<iframe src="../demo_zh/mbarrier_tma_timeline.html" title="mbarrier signalling TMA completion" loading="lazy"
        style="width:100%; height:700px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```

*交互式：TMA 加载通过 mbarrier 发出完成信号。MMA 路径在读取共享内存 tile 之前等待 barrier。Tensor Core 到 epilogue 的交接遵循相同的模式，只是由 Tensor Core 提交路径执行到达，而非 TMA。*

相同的思路也适用于资源重用。Barrier 不仅是数据就绪的信号。它也可以是"资源空闲"的信号。共享内存阶段在所有旧 tile 的消费者完成使用之前不能被覆盖。TMEM 区域在先前用户完成读取或写入之前不能被重用。在这些情况下，到达意味着"我已使用完此资源"，等待意味着"现在可以安全地为下一阶段重用此资源"。

这是理解流水线化 GEMM 内核中同步的正确方式。等待和到达并非作为防御性编程而散乱放置。每个都标记了一个具体的所有权转移：tile 变为就绪、累加器变为可读、或缓冲区变为可重用。一旦这些交接点被识别出来，控制流就变得更容易理解了。
