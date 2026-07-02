(chap_warp_spec_debug)=
# 调试 Warp 特化 Kernel

{ref}`chap_gemm_advanced` 中的 GEMM 第 7-9 步重叠了 TMA 加载、`tcgen05` MMA 和 TMEM/SMEM 写回。相同的调试方法适用于 Flash Attention 的握手：识别角色，识别每个角色拥有的存储，然后对照该模型验证生成的 CUDA。

不要从重写 kernel 开始。首先确保运行是有效的，然后检查生成的 CUDA。在排除环境和编译时问题后，这些 kernel 中的运行时失败通常归结为握手断裂：未初始化的 barrier、错误的到达计数、隐藏在角色守卫内的集体操作、过时的 barrier 相位，或在生产者使其写入可见之前就复用的存储。

## 在调试 Kernel 之前

首先排除运行时上下文：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

这些 kernel 面向 Blackwell (`sm_100a`)。如果 Python 导入了过时的 TVM checkout，或 GPU 不是 Blackwell 级别的，请在更改 kernel 之前修复。然后在查看性能之前运行 kernel 的最小正确性检查，例如 `run_correctness()`。

## 调试工作流

1. 在仍然失败的最小形状上复现失败。如果失败是非法内存访问，在下一次运行之前重启 Python。
2. 如果编译失败，在阅读运行时同步代码之前，检查已安装的 API、target、`dispatch=` 以及缓冲区范围。
3. 保存 `inspect_source("cuda")` 输出。在再次阅读 Python 之前，在其中搜索角色守卫、`mbarrier_init`、`tcgen05`、`cp.async.bulk.tensor` 和 `cta_sync()`。
4. 为失败的 kernel 路径编写角色/存储/握手/生命周期表。
5. 对照该表检查生成的 CUDA：barrier 初始化在角色分支之前，预期的 TMA 生产者，MMA 发出者，写回组，以及仅 warpgroup 分支内没有 CTA 范围的集体操作。
6. 将运行分类为死锁、崩溃、错误结果或正确但慢的运行，然后使用下面的匹配部分。
7. 一次更改一个握手：init 计数、arrive/wait 相位、角色守卫、fence、TMA 存储排空、TMEM alloc/dealloc 或 tile 调度器前进。
8. 在测量性能之前重新运行正确性检查。

## 传输内容

对于任何异步 kernel，在更改代码之前制作一个小的工作表：

| 项目 | 记录内容 |
|---|---|
| 角色 | 发出每个异步操作的确切线程、warp、warpgroup 或 CTA。 |
| 存储 | 每一步每个 tile 的活动位置：GMEM、SMEM、TMEM 或寄存器。 |
| 握手 | 生产者、消费者、信号对象、到达计数、相位，以及使数据可见的 fence 或排空。 |
| 生命周期 | 每个存储槽位可以复用、读回或释放的最早点。 |

然后对照工作表验证生成的 CUDA：

- 角色守卫与角色表匹配。
- Barrier 初始化出现在受守卫的角色分支之前。
- 集体操作没有被 lane、warp 或 warpgroup 守卫意外缩小范围。
- Arrive/wait 相位与握手表匹配。
- TMA 存储排空、TMEM dealloc 和 SMEM 复用仅在生命周期表说它们合法之后发生。

对 TMA→MMA→写回 GEMM 流水线以及 Flash Attention 中的分数/softmax/value/修正握手使用相同的工作表。

## 如果编译失败

在调试运行时同步之前修复编译时失败：

| 症状 | 可能的区域 | 首先检查 |
|---|---|---|
| 未知 TIRx API 或属性错误 | 安装的 wheel 与教程代码不匹配 | 打印 `tvm.__file__` 和 `tvm.__version__`；将 API 名称与 {ref}`chap_language_reference` 进行比较。 |
| 不支持的 `dispatch=` | 选定的 target 或原语不支持该路径 | 检查 `dispatch` 参数和 target 能力；本教程中的 `tcgen05` 路径需要 Blackwell。 |
| 缓冲区范围不匹配 | 缓冲区通过错误的硬件路径使用 | 检查工作表的存储行：TMEM 必须通过 `tcgen05` 访问，TMA 操作数必须使用兼容的 GMEM/SMEM 布局。 |
| 编译成功但生成的 CUDA 缺少预期路径 | 分发没有按预期降低 | 在更改算法之前检查生成的 CUDA 中是否有 `tcgen05` 和 `cp.async.bulk.tensor`。 |

## 检查生成的代码

对于任何编译后的 kernel，保存 CUDA 以便搜索和对比：

```python
from pathlib import Path

cuda_source = ex.mod.imports[0].inspect_source("cuda")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/my_kernel.cu").write_text(cuda_source, encoding="utf-8")
print(cuda_source)
```

生成的代码将 TIRx 构造映射到 CUDA，如下所示：

| TIRx | 生成的 CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` 内部守卫 | `((int)threadIdx.x) < 1`（仅 CTA 线程 0） |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

在阅读完整的 kernel 之前扫描这些字符串：

| 生成的 CUDA | 检查 |
|---|---|
| `if (threadIdx.x < 1)` | 单 CTA 线程守卫，通常是 barrier 初始化 |
| `mbarrier_init` | Barrier 初始化存在并出现在角色分支之前 |
| `tcgen05` | Tensor Core 路径已生成 |
| `cp.async.bulk.tensor` | 拷贝已降低为 TMA |
| `cta_sync();` | CTA 范围 barrier；绝不能位于 `wg_id` 分支内 |

## 第 7 步参考骨架

正确编译的第 7 步 kernel 具有以下顶层形状。下面的守卫为了可读性使用角色名称编写；在生成的 CUDA 中，搜索上表中对应的表达式。

```c
// (1) Barrier init：顶层，仅 CTA 线程 0
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // 由所有 128 个 WG0 线程到达
}

// (2) TMEM alloc：WG0 warp 0，发起 warp 的所有 lane
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) Fence + cta_sync，然后 phase init：producer=1, consumer=0

// (4) Warp 特化循环
if (wg_id == 1 && warp_id == 3 && elect_sync) { /* TMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 1 && warp_id == 0 && elect_sync) { /* MMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 0)                                { /* WB   */ while(valid){ ... next_tile(); } }

// (5) 清理：发起 warp，无 lane 守卫
cta_sync();
if (warp_id == 0) { tcgen05_relinquish_alloc_permit(); tcgen05_dealloc(..., 512); }
```

在更改算法之前检查这些：

- Barrier init 位于顶层，不在 `wg_id` 守卫内。
- `tcgen05_alloc` 和 `tcgen05_dealloc` 有 warp 守卫但没有 lane 守卫；发起 warp 的所有 lane 都参与。
- TMA 和 MMA 循环都迭代 `K_TILES` 次。
- Phase init 是 producer=`1`, consumer=`0`。

## 症状映射

从症状开始，但将其视为线索而非最终诊断：

| 线索 | 可能的区域 | 首先检查 |
|---|---|---|
| Kernel 挂起，然后运行时报告未指定的启动失败 | 死锁 | Barrier init 放置、到达计数、`cta_sync()` 放置和 `next_tile()` 参与 |
| 非法内存访问、XID，或后续无关 CUDA 调用也失败 | 崩溃/中毒上下文 | 重启 Python，然后检查指针范围、存储生命周期和集体操作参与 |
| 错误行出现在 128 行或 tile 大小的条纹中 | 同步竞争或 tile 索引不匹配 | 生产者/消费者相位、调度器前进，以及哪个 warpgroup 拥有每个行带 |
| `NaN` 或明显无效的值 | 描述符、操作数设置或未初始化的累加 | SMEM/TMEM 描述符设置、交错/布局和累加器初始化 |
| 有限但模式化的错误值 | 陈旧或部分可见的数据 | 缺少 fence、缺少 TMA 存储排空，或存储在生命周期表允许之前被复用 |
| 输出正确但没有预期的加速 | 分发或资源问题 | 生成的 CUDA 路径、流水线深度、占用率和寄存器溢出 |

## 何时重启 Python

CUDA 错误并不总是自行清理。在非法内存访问、XID 或"CUDA context poisoned"错误之后，后续无关的调用（如 `torch.randn`）可能会继续失败。在测试下一个修复之前重启 Python 进程，否则你可能在调试前一次崩溃而非当前代码。

## 死锁

按顺序检查这些：

- **到达计数与 init 计数不匹配。** 常见情况：`MBarrier.init(128)` 但 `arrive` 被 `if warp_id == 0: if lane_id == 0:` 守卫，所以只有 1 个线程到达，等待永远不返回。

  | Barrier | init(count) | 谁到达 | 到达数 |
  |---|---|---|---|
  | `TMABar` (tma->mma) | 1 | TMA 引擎通过 `arrive(stage, bytes)` | 1 |
  | `TCGen05Bar` (mma->tma, mma->ld) | 1 | MMA warp 通过 `tcgen05.commit` | 1 |
  | `MBarrier` (ld->mma) | 128 | 所有 WG0 线程通过 `arrive` | 128 |

- **Barrier init 嵌套在 `wg_id` 守卫内。** `.init()` 降低为 `if threadIdx.x < 1:`，意味着 CTA 线程 0。CTA 线程 0 位于 WG0，所以 `if wg_id == 1:` 阻止了任何线程运行 init。Init 必须在顶层；`grep mbarrier_init` 在 `inspect_source()` 中验证。

- **`cta_sync()` 在 warpgroup 分支内。** `cta_sync` 是 `__syncthreads()`，需要所有 CTA 线程。在 `if wg_id == 0:` 内部，WG1 永远不会到达它。使用 `T.cuda.warpgroup_sync(10)` 作为单 warpgroup barrier。

- **某些消费者 warpgroup 线程跳过了 `tile_scheduler.next_tile()`。** 调度器跟踪每个线程的状态；跳过它的线程可能永远循环。

- **TMA 和 MMA 对 K-tile 计数不一致。** 如果 MMA 做 `K_TILES - 1` 而不是 `K_TILES`，barrier 相位漂移，第二个外部 tile 死锁。

- **`PipelineState` 初始相位错误。** 生产者从 `phase=1` 开始，因此第一次等待通过；消费者从 `phase=0` 开始，因此第一次等待阻塞。如果两者从相同的相位开始，第一次握手可能立即死锁。

## 崩溃和上下文中毒

常见原因：

- **`pool.alloc` 在 `pool.commit()` 之后。** Barrier 包装器内部调用 `alloc`。正确顺序：`tmem_addr -> barrier wrappers -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()`。
- **带有 lane 守卫的 `tcgen05.alloc` 或 `tcgen05.dealloc`。** 发起 warp 的所有 lane 必须参与。`if lane_id == 0:` 只运行一个线程，这是未定义行为。
- **在 `tcgen05.dealloc` 之前缺少 `cta_sync()`。** TMEM 在写回仍在读取时被释放。
- **超出范围的 GMEM 或 SMEM 访问。** 缩小到一个 tile，检查调度器的 `m_idx`/`n_idx`，并检查当前形状是否是 kernel tile 或 cluster tile 的倍数。

## 错误结果

在猜测之前按模式分类错误输出。整行条纹通常指向生产者/消费者相位、tile 索引或角色所有权不匹配。`NaN` 输出通常指向描述符设置、操作数设置或未初始化的累加。有限但模式化的错误值通常意味着消费者读取了旧的 tile、部分写入的 tile，或存储尚未排空的数据。

- **`tcgen05.commit` 在 `elect_sync` 之外。** 所有 32 个线程创建 commit 组；31 个空组立即通知 mbarrier。TMA 可以在 MMA 读取之前覆盖 SMEM。
- **TMA 存储之前缺少 `fence.proxy_async("shared::cta")`。** TMA 引擎可能看不到线程的 SMEM 写入。
- **TMA 存储之后缺少 `cp_async.bulk.commit_group()` 加 `wait_group(0)`。** 下一个 tile 可以在存储排空之前复用 Dsmem。
- **持久化 kernel 在小规模（如 1024x1024）间歇性失败。** 较大的规模可以用更长的 K 循环掩盖竞争。重新检查 tile 之间的相位重置和 TMA 存储 commit/wait。
- **`fence.after_thread_sync()` 通常不是修复方法。** MMA 完成 mbarrier 已经携带释放-获取语义。第 8 步和第 9 步在写回边上保守地添加它，位于 `mma2ld.wait` 之后和第一个 `tcgen05.ld` 之前；不要在 TMA 到 MMA 的边上常规添加它。

## 正确但慢

如果输出正确但性能远低于预期，使用相同的检查循环：

| 线索 | 可能的区域 | 首先检查 |
|---|---|---|
| 生成的 CUDA 没有 `cp.async.bulk.tensor` | 拷贝没有降低为 TMA | 检查 `dispatch="tma"`、target 能力和操作数布局 |
| 生成的 CUDA 没有 `tcgen05` 路径 | MMA 没有降低为 Blackwell Tensor Core 指令 | 检查 `dispatch="tcgen05"`、target 能力和操作数布局 |
| TMA 和 MMA 不重叠 | 流水线太浅或相位将生产者/消费者串行化 | 检查生成的 CUDA 中 wait/arrive/advance 的顺序 |
| 小规模正确性好但大规模速度差 | 寄存器溢出、占用率或暂存缓冲区压力 | 检查编译器资源报告；减小 tile 大小、分块写回或降低流水线深度 |

## 提交一个好的 Issue

如果失败在上述检查中幸存，在 [Apache TVM GitHub 仓库](https://github.com/apache/tvm/issues) 提交 issue 之前先缩减它。包括：

- `tvm.__file__` / `tvm.__version__` 输出和 GPU 能力；
- 复现失败的最小形状；
- 失败是编译时、死锁、崩溃、错误结果还是正确但慢；
- 最小的 kernel 或 notebook 单元及其正确性检查；
- 保存的 `inspect_source("cuda")` 输出，或显示可疑守卫、barrier 或分发路径的最小摘录。
