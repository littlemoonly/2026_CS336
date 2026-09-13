### 3.1 Zero-Shot MMLU Baseline

MMLU 是常识数据集

使用 MMLU example 来构造 formatted task prompt（mmlu_zero_shot.prompt），之后这个 task prompt 被放入 **`zero_shot_system_prompt.prompt`** 中的 instruction 位置

使用 贪婪解码（Greedy Decoding）**temperature 0.0 and top-p 1.0**

### 3.2 GSM8K数据集

小学数学

task prompt : `gsm8k_zero_shot.prompt`

```
{question}
Answer:
```

formatted task prompt 作为 instruction 嵌入 `/zero_shot_system_prompt.prompt`

### 3.3 AlpacaEval

#### Problem (alpaca_eval_baseline)：4 pts

### 3.4 SimpleSafetyTests

### 4.1 Looking at Instruction Tuning Data

`train.jsonl.gz` ???

每一行都是JSON对象，包括 `prompt` 和 `response` key

### 4.2 Implementing Instruction Fine-Tuning

这个是 prompt 模版

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}

```

#### dataloader

实现一个Dataset 的子类：

documents列表是不同长度的str，具体是 prompt_template 填入prompt和response后的str

这些 str 拼在一起形成大的 token_ids，它们被切分成定长（seq_len）的序列

eg. input: I 	love ML eon

label :	love  ML  End 没了

所以只能形成 `len(input) - 1`对

Dataloader 返回 Iterable

流程：对于 map-style Dataset，DataLoader 大致执行：

```
 Dataset Sampler 生成索引顺序 -> 按照 batch_size 将索引分组 -> 调用 dataset[i] 读取每条样本->collate_fn 将多条样本堆叠->返回一个 batch
```

#### training script

每次按照当前 accumulation window 的大小对 loss 进行 scale，达到多个microbatch累积，跟一次计算整个batch相同的结果

```
scaled_loss.backward()  # 计算并累加 grad
```

