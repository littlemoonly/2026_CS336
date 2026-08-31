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

