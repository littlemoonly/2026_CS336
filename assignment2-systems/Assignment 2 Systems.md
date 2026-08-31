

[toc]

## **Assignment 2: Systems**

## 2. Profiling(性能分析) and Benchmarking(基准测试)

### 2.1.3 End-to-End Benchmarking 端到端基准测试

#### Problem (`benchmarking_script`) : Benchmarking Script (4 points)

##### with warmup

1. 纵轴采用了对数坐标，因此1k到10k不是均匀的
2. 每根柱子顶部的**黑色短横线**是误差线，表示平均吞吐量±1个标准差，计算方法是**误差传播**近似得到吞吐量标准差$σ_{throughput}≈ mean_{sec} / std_{sec}$

![image-20260803111623104](/Users/xinyue/Library/Application Support/typora-user-images/image-20260803111623104.png)

`xl` 和 `10b` 的 `model_config` 在我的V100上OOM了

##### without warmup

为什么第一次执行更慢

- GPU 上的第一次 forward、backward 或 optimizer step 不只是执行模型计算，还可能包含以下额外工作：CUDA 上下文初始化、CUDA kernel 的延迟加载、GPU 内存分配和缓存建立等
- 另外，一个 warm-up step 不一定足以完成所有初始化工作

#### Problem (`nsys_profile`): Nsight Systems Profiling (5 points)

这部分没做，要求我们用 Nsight system 更细致地profile我们的代码

NVTX 是 NVIDIA Tools Extention，我们用NVTX range 在程序时间线上添加带名字的区间。

#### `torch.cuda.synchronize()` 

由于CPU向GPU提交任务后就会返回，CUDA 是异步执行的。因此需要用 `torch.cuda.synchronize()` 保证所有GPU任务都已经完成

#### CUDA kernel

常见Kernel分类

**GEMM (Matrix Multiply) Kernel**：矩阵乘法，计算密集型（Compute-bound）

**Element-wise Kernel**：逐元素运算，如 `ReLU`、`Add`、`Cast`。

**Reduction Kernel**：规约运算（多维变一维），如 `Sum`、`Mean`、`LayerNorm`。

forward 中最耗时的通常是某种**矩阵乘法 GEMM kernel**，其他比较耗时的kernel还有 softmax，RMSNorm，SiLu/gating等

相比只forward，加上反传和优化参数，矩阵乘法所花时间占比应该变少了，因为反传和优化器引入了更多 Element-wise, Reduction Kernel

Softmax 通常比 Attention 中的矩阵乘法快得多，但两者运行时间的差距远小于它们 FLOPs的差距。 这是因为矩阵乘法经过了高度优化且属于计算密集型（Compute-bound）；而 Softmax 的算术强度（Arithmetic Intensity）较低，通常受限于内存带宽（Memory Traffic）、规约操作（Reductions）以及 Kernel 发射开销（Kernel-launch Overhead）

#### 混合精度训练

- **FP16：**有效数字相对较多，但可表示的数值范围较小，容易出现梯度下溢为 0 或数值上溢为 `NaN`，所以通常需要 loss scaling。

  FP16 只有 10 位尾数（约 3-4 位十进制有效数字），会导致**严重舍入误差**，使方差估计不准确。

- **BF16**（brain floating point）：wide dynamic range/动态范围大，有效数字较少，但指数位与 FP32 相同，因此数值范围与 FP32 接近

- **FP32**

用 FP16/BF16 加速矩阵乘法；

用 FP32 完成容易积累误差的求和、归约和部分归一化；

通过 `torch.autocast` 自动选择合适精度；

在 FP16 训练中通常再结合 loss scaling。

#### Problem (`mixed_precision_accumulation`) 混合精度累加

- 首先，输入精度和累加精度都是fp32，结果误差很小：

```python
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
# 10.0001
```

- 输入精度和累加精度都是fp16，1000 个 0.01 加起来，用 fp16 数值+fp16累加，最后的误差已经是0.05了，已经算挺大的误差了

原因是当 `s` 接近 10 时，虽然你希望每次增加约 `0.01`，实际存储值只能在 FP16 的离散点之间跳动

```python
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
# 9.9531
```

- 输入量化为fp16，但累加保持fp32，误差稍小

与 FP32 的 `s` 相加时，PyTorch 会将右侧值提升到 FP32 再进行计算。但提升精度只能准确保存当前 fp16 的值

```python
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
# tensor(10.0021)
```

因此，混合精度计算中经常采用**低精度输入或乘法，FP32 累加**

#### Problem (`benchmarking_mixed_precision`)

```python
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
```

首先，如何理解 ToyModel（Post-Norm，LayerNorm是在特征维度内部做）

其次，分析`forward`每一步是什么数据类型

在 PyTorch 的 `torch.cuda.amp.autocast(dtype=torch.float16)` 下，PyTorch 维护了两个调度表：

| 类别          | 包含的 Op                              | 行为                                 |
| ------------- | -------------------------------------- | ------------------------------------ |
| **FP16 列表** | `nn.Linear`、卷积、`matmul`、LSTM 等   | 计算和输出均为 FP16                  |
| **FP32 列表** | `nn.LayerNorm`、`Softmax`、Loss 函数等 | 始终以 FP32 计算和输出（数值稳定性） |

对于两个列表都不在的 Op（如 ReLU），按输入 dtype 原样执行。



对于 LayerNorm，求和、方差，以及分母的小常数 $ε$ 都对精度敏感，因此要用 `fp32`。

1. 具体的，`fp16`最大值仅 ~65504，甚至不支持d=1000，每个元素100的向量求和。
2. 方差计算的 `x_i - μ`  很小，导致 Catastrophic cancellation，`fp16`只有3-4位十进制小数，会产生严重的舍入误差

但是，如果用 `bf16`，就无须单独处理 LayerNorm，因为BF16和fp32一样，都有8位尾数，动态范围较大。而LayerNorm对混合精度敏感的主要原因是 fp16 求和溢出；而 Catastrophic cancellation 的误差可以接受，因为 LayerNorm 的目的是稳定激活值的分布（均值和方差的尺度），而不是需要极高精度的逐元素计算。



#### Problem (`memory_profiling`)

#### 解读`ctx=1024, size=m, fbo, amp=T`的memory timeline

![image-20260801103918055](/Users/xinyue/Library/Application Support/typora-user-images/image-20260801103918055.png)

上图解读:

完整训练的 active-memory timeline 一般可以分成三段：

**Forward：**
 显存逐层上升，因为各层 activation 被保存下来。长 context 下，这一段通常产生最大峰值。

**Backward：**
 **时间线通常呈现下降的锯齿。**反向传播从最后一层开始，每处理完一层，该层保存的 activation 会被释放；与此同时会生成参数梯度。因此它不是平滑下降，而是边释放 activation、边产生 gradient。

**Optimizer step：**
 最后会出现一个较窄的峰或者一段新的高平台。AdamW 为每个参数保存一阶矩 $m$ 和二阶矩 $v$，两者大小都和参数相同。第一次 `optimizer.step()` 后，这些状态不会被释放，因此基础显存平台会永久升高。

#### 解读 `ctx=2048, xl, f, amp` 的memory timeline

![b844064b-67cc-45da-8c0a-c42f28c0f8fe](/Users/xinyue/Downloads/b844064b-67cc-45da-8c0a-c42f28c0f8fe.png)

每个entry代表一次显存事件

**基线显存**约19GiB, 包括参数和优化器的12.7GiB,以及6GiB的 `bf16`副本(`autocast` context 很可能包住了整个 profiling 循环，因此 BF16 权重缓存一直存活),加起来是19

**峰值显存**约为25GiB, 最高峰主要由形状为 $(B,H,T,T)$ 的 attention score 和 softmax 张量导致。

锯齿来自临时 activation 和算子 workspace 的反复申请与释放

#### [显存分析](https://chatgpt.com/g/g-p-6a3aa14bd8e48191a8430bc425f2a013-cs336/c/6a6d58e7-d6d4-83ee-bd38-6d50d616dc64)

在 batch size 4、vocabulary size 10,000、FP32、朴素 attention 且不使用 checkpointing 的情况下，context length 128 的 forward 峰值约为 **18.0 GiB**，完整训练步骤约为 **51.4 GiB**

> [!NOTE]
>
> 如何估算forward 和 full train 的显存峰值
>
> 对于 ctx=128, 激活不够大, 显存估算主要是**参数和优化器**(参数+梯度+m,v状态 = 4*参数量)
>
> 例如, xl , ctx = 128, 参数量 `P = 2Vd+L(4d^2+3ddff+2d)+d≈3.407×10^9`个, **P=12.7GiB**乘4得到完整训练峰值约50G
>
> 但是对于ctx=2028, forward 后激活值(attention score + softmax后weight)很大,估计峰值分别约为 **216.5 GiB** 和 **242 GiB**，因此在常见的 80 GiB GPU 以及 192 GiB B200 上都会 OOM；其显存增长主要来自 $B\times H\times T\times T$ attention 张量的二次方增长。

**(c) **  **Mixed precision 的峰值显存**: **观察使用 amp 之后,`xl` 模型的 peak memory usage 在 f 和 fbo 模式的变化**

ctx=128时, 显存主要是参数、梯度和优化器状态, 他们在amp时依然以`fp32`存储,而且可能缓存一份 BF16 权重副本,所以短 context 的 forward 显存甚至可能比 FP32 inference 略高。

但在 context 2048 时，**注意力矩阵和 saved activations 很大**，BF16 将这些张量的大小近似减半，因此训练峰值会明显下降。

**(d)** **`xl` 下 残差流激活张量的大小( residual-stream activation tensor )**

<img src="/Users/xinyue/Library/Application Support/typora-user-images/image-20260801112433273.png" alt="image-20260801112433273" style="zoom:50%;" />

形状 [B, T, d] , 大小 `4Bytes * B*T*d`

所以：

$T=128:\quad 0.0390625\times128=5\text{ MiB},$

 $T=2048:\quad 0.0390625\times2048=80\text{ MiB}.$

**(e) `xl` 在forward时, 查看最大的allocation**

![image-20260801114638836](/Users/xinyue/Library/Application Support/typora-user-images/image-20260801114638836.png)

当然是 `scaled_dot_product_attention` 函数里计算 $QK^\top$、mask 或 softmax 的代码, 图中非蓝色彩色的那些全是大小为2GiB的内存块——因为因attention scores、mask 后的 scores、softmax probabilities 等中间结果都可能具有相同的  `(B,H,T,T)=(4,32,2048,2048)`形状。

**(f)** 观察目标是 **residuals / saved tensors**, 一个 `TransformerBlock`中forward 结束时仍然存活、直到该 block backward 才释放的张量. 每个 Block 的 residual 的计算方法如下

- Mstart：这个 block forward 进入时的 active memory；
- $M_{\text{end}}$：这个 block forward 返回时的 active memory。

那么：$R=M_{\text{end}}-M_{\text{start}}$ 就是该 block 新增长期保留到 backward 的内存近似值

**五个最大贡献项应该是什么量级**

1. **Attention张量,前两名**: 通常会有两个长期存活的约 **2048 MiB** 张量：一个是 softmax 的 exponentiated intermediate，另一个是供 $PV$ 和 backward 使用的 attention probabilities(???)
2. **SwiGLU 张量, 多个并列**

**在 backward pass 中, 这些residual会被free,但会有新的grad张量产生, 那么如何计算新产生的 grad 张量大小呢?**

Mdrop = **backword**前后 activate memory 的差值

Mdrop = R - G, 记G为新产生的grad大小, Mdrop和R都是可观测的,从而计算出G

直接估算G, 应该为每个 Block所有参数大小 P
$$
\begin{aligned}
P_{\text{block}}
&=4d_{\text{model}}^2
 +3d_{\text{model}}d_{\text{ff}}
 +2d_{\text{model}}\\
&=4(2560)^2
 +3(2560)(10240)
 +2(2560)\\
&=104{,}862{,}720.
\end{aligned}
$$
FP32 中每个参数的 gradient 占 4 bytes，所以：
$$
\begin{aligned}
G_{\text{expected}}
&=\frac{104{,}862{,}720\times4}{1024^2}\\
&=\boxed{400.02\text{ MiB}}.
\end{aligned}
$$

## 3 Single-GPU Memory

### 3.1 Autograd Residuals

#### 什么是 Residuals

为反向传播保存的张量叫 saved tensors / autograd residuals / 激活残差

#### 什么是 eager PyTorch 

####  Operation Fusion(算子融合)

核心是把原本由多个细粒度 PyTorch 算子组成的 RMSNorm，视为一个整体来执行和求导。

通过 `torch.compile`, PyTorch 不再让 autograd 分别处理 `pow`、`mean`、`rsqrt`、`mul` 等节点, 而是把整个 RMSNorm 当成一个复合函数, 只需要保存反向传播真正必需的张量。

#### torch.compile()

```
block = torch.compile(block, fullgraph=True)
```

它的目标是让编译器看到整个 TransformerBlock 的 forward 计算图，并尽可能进行: operator fusion, kernel fusion, 中间张量消除、内存复用, forward / backward 联合优化

##### `pack_hook`

**调用时机:** `pack_hook` 在 forward 中，每当 autograd 决定“这个张量以后 backward 要用”,因此保存张量时被调用

### 3.2 Gradient Checkpointing

#### Problem (`gradient_checkpointing`)

**(a) 给定一个有 $N$ 个相同 Transformer Block 的模型, 我们想要找到一个 最小化 peak activation memory 的checkpoint方法**, 也就是**忽略计算成本，怎样使峰值内存最小**

首先, 如果像下面每个块执行 checkpoint, 它会长期保存 $N$ 个 block 的入口激活，所以 checkpoint memory 仍然是 $O(N)$

```
for _ in range(N):
    x = checkpoint(block, x)
```

真正的最小内存方案是构造一个**左偏的递归嵌套 checkpoint**(因为backward是从后往前传的)

把前 $n-1$ 个 block 作为一个 checkpoint，再正常执行第 $n$ 个 block。backward 时先重算前 $n-1$ 层得到第 $n$ 层输入，只保留第 $n$ 层的 residuals；第 $n$ 层 backward 完成后，再递归处理前 $n-1$ 层。

**（b）只允许一层 recomputation，如何选分组大小**
$$
M_{\text{peak}}(k)
\approx
\frac{N}{k}A+kR
$$
立刻得到 $\boxed{
k^*
\approx
\sqrt{\frac{NA}{R}}
}$, 如果简单假设 $A$ 和 $R$ 同阶，就得到经典结论：
$$
k^*\approx\sqrt N,
\qquad
M_{\text{peak}}=O(\sqrt N)
$$
对于本题,N=32, A=每个ckp边界激活=size of input x = 80MiB, $R\approx3651.31\text{ MiB}$ (教程测得一个编译后的 block 保存的Residuals), 得k=0.84,所以k=1最优

## 4 GPU Kernels

### Problem (`pytorch_attention`)

基线内存( `baseline_memory`)  包括 `Q,K,V, grad_output`, 不包括计算图或者input的梯度

**反向前的峰值内存**(`memory_before_backward_bytes`)是 forward 在构造了计算图,保存激活张量后,就要执行反向传播的内存用量

保存激活值内存(`saved_delta_mib`)是上面两者之差, 是autograd为了反传而存储的张量

结果已经保存在 `attention_benchmark.csv`

![image-20260802133501550](/Users/xinyue/Library/Application Support/typora-user-images/image-20260802133501550.png)

重点看 `saved_delta_mib` 随着 seq_len 和 dmodel 的变化.

1. 随着序列长度 `seq_len` 从 256 增加到 8192（增加 32 倍），保存的激活值内存（`saved_delta_mib`）从 ~4.15 MiB 暴增至 ~4100 MiB（增加约 **1000 倍**），直观体现了 Attention O(N2) 的空间复杂度
2. 增大 `d_model` 对耗时和内存有增加作用但很小



### 4.2 Benchmarking JIT-Compiled Attention

### Problem (`torch_compile`)

**(a) profile compiled `scaled dot product attention`**

绿色部分和没有 compile 之前基本完全一样, 红色部分 前向和反向的时间明显减小了

结论: 

![image-20260803105515373](/Users/xinyue/Library/Application Support/typora-user-images/image-20260803105515373.png)

**(b) profile entire `Transformer Model`**

forward 和 backward 一般会加速, 加入 optimizer step 后, 整体加速比例会下降, 因为 `compiled_model = torch.compile(model)`通常编译模型的 forward 和 由此派生的 backward 图, `optimizer.step()` 一般仍在编译模型之外进行

这就引入了下面的 triton 

#### 4.2.1 Example - Weighted Sum

#### program instance

一个 program instance 可以理解为执行同一段程序的一个 GPU 线程块

两个 tile 维度:

- `ROWS_TILE ` 表示一个 program instance 一次负责多少行
- `D_TILE` 表示每次沿着特征维度 $D$ 加载多少列

##### `tl.make_block_ptr()` 函数

返回的是一个 **block pointer 描述符**

参数含义如下: 

```python
x_block_ptr = tl.make_block_ptr(
  x_ptr: 整个原来tensor首元素的指针
  shape: 整个原来tensor的形状
  
  strides: 每沿着一个维度移动一个逻辑元素时, 内存地址需要移动多少个元素
  offsets: 表示当前 block 左上角在原始 Tensor 中的逻辑坐标; 注意是逻辑, 不需要手动乘 stride
  block_shape: 表示这个 block pointer 一次描述或加载的 tile 形状
  order:一般是(1,0),表示最后一维是连续的
)
```



#### `WeightedSumFunction(torch.autograd.Function)`

##### 什么是 `grid`

它规定要启动多少个 Triton program instance, 例如 `weighted_sum` 中, 启动**一维任务网格** :

```python
grid = (triton.cdiv(n_rows, ROWS_TILE_SIZE), )	
# 相当于grid = (program_num, ), 其实是tuple[int]
```

##### 什么是 `kernel` 和 program instance

kernel 是“程序模板”，program instance 是这个模板的一次具体执行

Triton 会按照 grid 规定的形状, 启动多个 program instance, 每个 intance 都执行相同的 kernel 代码, 拥有不同的 `program_id`
$$
\text{一个 kernel 定义}
\quad\xrightarrow{\text{launch grid}}\quad
\text{多个 program instances}
$$

##### 函数流程

功能: 负责把刚刚的 weighted_sum_fwd 包装

整理形状
用 `ctx.save_for_backward()` 保存反向传播所需的张量
launch grid

#### `weighted_sum_backward()` 函数

##### 函数流程

先用 `make_block_ptr()` 得到 输入, 输出的 block_ptr

之后沿着 D 维度循环处理D TS:
首先, 算grad_x, load所需, store 结果
之后, 算grad_w,  
移动所有的block_ptr

### 4.2.2 FlashAttention-2 Forward Pass

𝑷 (attention scores) 形状 `(batch_size, n_heads, seq_len, seq_len)` 所以每多一个头, 内存都会翻倍?
