# train&tuning

CS336 assignment1 的训练和调参

### Problem (batch_size_experiment)

batch size 较小时，GPU 上的矩阵乘法规模较小，GPU 可能没有被充分利用。因此通常会出现 $B\uparrow
\quad\Rightarrow\quad
\text{tokens/s}\uparrow$

**critical batch size** 或收益饱和点：超过该规模后，增加 batch 的主要收益只剩硬件并行效率，而不再显著改善优化效果。

#### 指标 ： `tokens_seen`

$tokens\_seen = BatchSize * ContextLength * train\_steps$

## Ablation

### Ablation1 · Problem (pre_norm_ablation)

> [!TIP]
>
> 注意Norm有两个选项：
>
> **Norm 放在哪里**：Pre-Norm 还是 Post-Norm。
> **Norm 用什么公式**：LayerNorm 还是 RMSNorm。

原始 Transformer 使用的是 **Post-Norm LayerNorm**，作业为了控制变量，仍然使用 RMSNorm，只改变 RMSNorm 的位置，因此实际比较的是：$\text{Pre-RMSNorm}
\quad\text{vs.}\quad
\text{Post-RMSNorm}$

对于现代 decoder-only 大语言模型，最常见的组合是 ${\text{Pre-Norm}+\text{RMSNorm}}$，因为他保留了更直接的残差梯度通道，训练更稳定

### Ablation 2 · position embeddings

### Ablation 3: SwiGLU vs. SiLU