## Assignment5

### Problem (prompting_baselines): Run OLMo-2-0425-1B on GSM8K (5 points)

> (a) Write a script to evaluate OLMo-2-0425-1B performance on GSM8K with zero-shot `question_only`, zero-shot `r1_zero,` and few-shot `r1_zero_three_shot` prompts. Then, run your script and observe the outputs. For each prompt, how many model generations fall into each of the following categories: (1) correct with both format and correctness reward 1, **(2) format reward 1 and correctness reward 0**, **(3) format reward 0 and correctness reward 0**? Observing at least ten examples of category 2, how many model outputs are actually correct but just not parsed properly? What about category 3?
>
> Deliverable: A few sentences of commentary, the evaluation metrics, and a few examples of prompts and responses.

*Ideal case:*

| Prompt               | (1) format=1, correct=1 | (2) format=1, correct=0 | (3) format=0, correct=0 |
| -------------------- | ----------------------- | ----------------------- | ----------------------- |
| `question_only`      | 6 (0.45%)               | 406 (30.78%)            | 907 (68.76%)            |
| `r1_zero`            | 0 (0.00%)               | 516 (39.12%)            | 803 (60.88%)            |
| `r1_zero_three_shot` | 233 (17.66%)            | 986 (74.75%)            | 100 (7.58%)             |

**For category 2,** 0/10 were actually correct. The parser successfullly parsed the answer, but the underlying reasoning were wrong.

**For category 3,** 2/10 were rejected because they did not follow the required format, but the output is numerically right.

> (b) Observing the model outputs, characterize the model’s behavior with each prompt. For example, if we want the model to answer the question, is it enough to just provide the question, or does the model exhibit other behaviors besides just answering the question? How do the zero-shot `r1_zero` and few-shot `r1_zero_three_shot` prompts shape the model’s behavior?
>
> Deliverable: A few sentences of commentary with supporting examples.

**For `question_only` prompt**, the model answers the problem directly, but it can also continue the text in a less task-focused way. **It means:** the pretrained base model does not consistently interpret a bare question as an instruction.

**For `r1_zero` prompt**, output can be malformed, or contain reasoning mistakes, although the required format is followed.

**For `r1_zero_three_shot` prompt**,  the generations look like complete GSM8K solution. **It means** prompting can substantially steer a base model' s behaviour.



### Problem (baseline_calcs)

> For (a) and (b)

No baseline: $Var = \frac1n p(1-p)^3$

with baseline: $\operatorname{Var}(\hat g_b) = \frac1n p(1-p)(1-p-b)^2$, the variance declines.

> (c) What is the resulting variance if we substitute the “population mean” baseline 𝑏 = 𝑝?Compare this variance to that of the unadjusted policy gradient estimator: is it always lower, always higher, or sometimes higher or lower depending on 𝑝?

Substituting $b=p$, the variance becomes: $\operatorname{Var}(\hat g_{b=p})=\frac1n p(1-p)(1-2p)^2.$

Compared with the unadjusted variance $\frac1n p(1-p)^3$, their difference has the same sign as $p(3p-2)$. Thus, **the population-mean baseline reduces variance for $p<2/3$, gives equal variance at $p=2/3$, and increases variance for $p>2/3$.**



#### grad clipping

a handwritten version:

```python
# gradient clipping before optimizer.step()
grad_norm_sq = 0.0
for p in model.parameters():
    if p.grad is not None:
        grad_norm_sq += p.grad.detach().norm(2).square()

grad_norm = grad_norm_sq.sqrt()	# first sum of squared, then sqrt

if max_grad_norm is not None and grad_norm > max_grad_norm:
    scale = max_grad_norm / (grad_norm + 1e-6)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(scale)
```

or:

```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```



### Problem (think_about_length_normalization)

> Before running any experiments, think about the difference between normalizing each sequence by sequence length, versus normalizing all sequences by the same constant. What are the pros and cons of each approach? Are there specific settings or examples where one approach seems better?

**normalizing by seq_len:**

效果：会让长 sequence 中的 token 相对于短 sequence 的 token 被 downweight，人为降低长回答中每个 token 的权重。

- Pros: preventing long generations from dominating the update and potentially reducing variance. 
- cons: changes the relative weighting implied by the standard sequence-level policy gradient

**more fits in :** *sequence lengths vary greatly* and equal *example* weighting is desired

**normalizing by same constant (DR.GRPO):**

- pros: better aligned with optimizing expected sequence reward
- cons: long responses can contribute larger gradients and increase variance.

**more fits in :** we want a less biased estimator of the *sequence-level* reward objective.

### Problem (think_about_rft)

> [!NOTE]
>
> 这题主要比较 RFT 和 Dr. GRPO 的两个差异：**是否使用 baseline，以及错误回答是否产生梯度**
>
> 直观上最重要的例子是：
>
> - 一组全错：RFT = 0，Dr. GRPO = 0。
> - 一组有对有错：RFT 只强化正确答案；Dr. GRPO 同时强化正确答案、压低错误答案。
> - 一组全对：RFT 仍然继续模仿这些正确答案；Dr. GRPO 因为 $r_j-\mu=0$，完全不更新。
>
> 因此 **Dr. GRPO 的 baseline 通常能减少梯度方差**，并把训练资源更多**集中到“有区分度”的 prompt 上**；RFT 更简单，而且即使一个 prompt 已经全部答对，仍会继续强化成功轨迹，这在希望持续模仿高质量正确解法时可能有用。

For binary rewards, RFT is equivalent to **applying the REINFORCE gradient** only to successful rollouts, so in expectation it follows the expected-reward policy gradient. Dr. GRPO subtracts the group-mean reward, giving an expectation scaled by $(G-1)/G$ relative to RFT (for the same constant normalizer), while usually reducing variance through centering.

RFT only reinforces correct responses, whereas Dr. GRPO also downweights below-average responses and gives zero update when all responses in a group have the same reward.

### Problem (derive_difficulty_reweightings)

https://chatgpt.com/g/g-p-6a3aa14bd8e48191a8430bc425f2a013-cs336/c/6a97c70b-8bb8-83ee-981f-e073b9230c27

### Problem (think_about_advantage_normalization)

> [!NOTE]
>
> 这个问题是在比较三种 advantage normalization 对不同 prompt 的梯度权重影响：
>
> - 除以 group std：让不同组的更新尺度更接近，训练可能更稳定，但会改变原始 expected reward 的优化目标，而且会放大 std 很小的组。
> - 除以 group mean：MaxRL 的做法，会更强调平均 reward 较低的困难 prompt；优点是把训练资源集中到难题，缺点是当 mean 很小时权重可能很大、导致不稳定。
> - 不做 normalization：Dr. GRPO 的做法，更忠实于原始 expected reward objective，但不同 prompt 的梯度尺度可能差异较大，训练稳定性可能较弱。

### Problem (grpo_train_step_variants_on_policy)

> [!CAUTION]
>
> TODO 还未实现
> conda run pytest -k test_grpo_train_step_variants_on_policy

### Problem (grpo_experiments_variants_on_policy)

| 方法          | baseline | advantage normalizer | loss normalization | 主要变化                           |
| ------------- | -------- | -------------------- | ------------------ | ---------------------------------- |
| Standard GRPO | mean     | std                  | sequence           | 原始基线                           |
| GRPO_constant | mean     | std                  | constant           | 只去掉 sequence normalization      |
| Dr_GRPO       | mean     | none                 | constant           | 再去掉 advantage std normalization |
| RFT           | none     | none                 | constant           | 只强化正确 rollout                 |
| MaxRL         | mean     | mean                 | constant           | 用平均 reward 归一化 advantage     |

实验其实是在逐项回答：

- `GRPO → GRPO_constant`：sequence normalization 到底有没有帮助？
- `GRPO_constant → Dr_GRPO`：除以 group std 到底有没有帮助？
- `Dr_GRPO → RFT`：减去 group-mean baseline 是否有价值？
- `GRPO_constant → MaxRL`：用 mean normalization 替代 std normalization 是否更好？



#### 6.3 GSPO

#### Problem (think_about_importance_reweighting)

**No reweighting**：完全不修正 rollout 来自旧策略 $\pi_0$ 这件事，所以当 $\pi_\theta$ 和 $\pi_0$ 差得较远时 bias 最大；但因为没有 importance ratio，variance 最小。

**PPO/GRPO token-level clipping**：只修正当前 token 的分布差异，没修正 prefix 和 suffix，因此仍有 bias；但 token-level ratio 加 clipping 比完整 sequence reweighting 稳定，属于中间方案。教程明确指出 token-level reweighting 忽略 prefix/suffix，而 sequence-level 可以修正这些 bias，但代价是更高 variance。

**GSPO**：用整条序列的 importance weight，因此更好地考虑整条 trajectory 的 distribution shift，通常 bias 更小；不过 sequence-level reweighting variance 更大，所以又通过 geometric mean 和 clipping 压低 variance，同时重新引入一些 bias。
