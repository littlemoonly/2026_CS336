from typing import Callable, Literal

import torch


def get_device(device: str = None) -> str:
    if device is not None:
        return device
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.optim import Optimizer


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt/response pairs for causal-LM training.

    Returned tensors are shifted so that input_ids[:, t] predicts labels[:, t].
    """
    prompt_tokens = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_tokens = tokenizer(output_strs, add_special_tokens=False)["input_ids"]

    prompt_lens = [len(tokens) for tokens in prompt_tokens]
    joined_tokens = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(prompt_tokens, output_tokens, strict=True)
    ]

    max_len = max(len(tokens) for tokens in joined_tokens)
    pad_id = tokenizer.pad_token_id
    # construct padded tensor: tokens + pad_token for each row
    padded = torch.full((len(joined_tokens), max_len), pad_id, dtype=torch.long)
    for row, tokens in enumerate(joined_tokens):
        padded[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    # joined_tok:How are you? Fine. prompt_len=3
    # padded:    How are you? Fine. pad pad 
    # input_ids: How are  you?  Fine. pad
    # labels:    are you? Fine. pad   pad 
    # resp_mask: F,  F,   T,    F,    F
    input_ids = padded[:, :-1]
    labels = padded[:, 1:]

    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    for row, (prompt_len, tokens) in enumerate(zip(prompt_lens, joined_tokens, strict=True)):
        seq_len = len(tokens)   # joined_tok, input+label拼接在一起的tokenid,不加padding
        # TODO(STUDENT): implement the core logic here.
        # Set response_mask[row, ...] to True exactly where labels[row, ...]
        # corresponds to an output token. Remember labels are shifted left by
        # one relative to joined_tokens, and padding positions must stay False.
        start = prompt_len - 1
        end = seq_len - 1
        response_mask[row, start:end] = torch.ones((end-start), dtype=torch.bool)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Score rollout responses and summarize reward components."""
    assert len(rollout_responses) == len(repeated_ground_truths)

    raw_rewards: list[float] = []
    format_rewards: list[float] = []
    answer_rewards: list[float] = []

    for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True):
        reward_parts = reward_fn(response, ground_truth)
        format_rewards.append(float(reward_parts["format_reward"]))
        answer_rewards.append(float(reward_parts["answer_reward"]))
        raw_rewards.append(float(reward_parts["reward"]))

    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
    format_rewards_tensor = torch.tensor(format_rewards, dtype=torch.float32)
    answer_rewards_tensor = torch.tensor(answer_rewards, dtype=torch.float32)

    metadata = {
        "reward_mean": float(raw_rewards_tensor.mean().item()),
        "format_reward_mean": float(format_rewards_tensor.mean().item()),
        "answer_reward_mean": float(answer_rewards_tensor.mean().item()),
    }

    return raw_rewards_tensor, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Normalize flattened rollout rewards within each prompt group."""
    rollout_batch_size = raw_rewards.shape[0] #  raw_rewards is flattened, shape [rollout_batch_size, ]
    assert raw_rewards.ndim == 1
    assert rollout_batch_size % group_size == 0

    grouped_rewards = raw_rewards.reshape(-1, group_size)   # shape: prompt_num , group_size

    # Inputs available:
    # - baseline == "mean": subtract each row's mean reward from that row
    # - advantage_normalizer == "std": divide by each row's std plus advantage_eps
    # Output expected:
    # - advantages: same shape as grouped_rewards, then flatten back to
    #   shape (rollout_batch_size,)
    advantages = grouped_rewards.clone()    # shape (n_prompts_per_rollout_batch, group_size)
    if baseline == "mean":
        grouped_mean_rewards = grouped_rewards.mean(dim=-1, keepdim=True)
        advantages -= grouped_mean_rewards
    elif baseline == "none":
        pass

    if advantage_normalizer == "std":
        grouped_std = torch.std(grouped_rewards, dim=-1, keepdim=True)
        advantages /= (grouped_std + advantage_eps)
    elif advantage_normalizer == "none":
        pass
    elif advantage_normalizer == "mean":
        grouped_mean_rewards = grouped_rewards.mean(dim=-1, keepdim=True)
        advantages /= (grouped_mean_rewards + advantage_eps)

    advantages = advantages.reshape((-1,))

    metadata = {
        "reward_mean": float(raw_rewards.mean().item()), # why not advantages?
        "reward_std": float(raw_rewards.std().item()),
    }
    return advantages, metadata # after normalization, rewards -> advantages


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none","noclip","grpo","gspo"] ="none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''function usage:
    taking the gradient (of this pgloss), will produce each term of the GRPO gradient estimator 
    params:
    raw_rewards_or_advantages:  Shape (batch_size,),or (batch_size, 1)
    policy_log_probs:           Shape (batch_size, sequence_length), logprobs for each token.
    How are you? Thank you. <pad>
        0.6  0.7  0.4  0.3   0.7 (unlogged)
    returns:
    per_token_policy_gradient_loss : Shape (batch_size, sequence_length)
    metadata
    '''
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)

    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -raw_rewards_or_advantages * policy_log_probs # shape (batch_size, sequence_length)
    elif importance_reweighting_method == "noclip":
        assert old_log_probs is not None
        ratio = torch.exp(policy_log_probs - old_log_probs)
        # unclipped objective J = A * w_t
        per_token_policy_gradient_loss = -raw_rewards_or_advantages * ratio
    elif importance_reweighting_method == "grpo":
        assert old_log_probs is not None and cliprange is not None
        ratio = torch.exp(policy_log_probs - old_log_probs)
        clipped_ratio = ratio.clamp(1 - cliprange, 1 + cliprange)
        # GRPO clipped objective J = min(A * w_t, A * clip( w_t))
        per_token_policy_gradient_loss = -torch.minimum(
            raw_rewards_or_advantages * ratio,
            raw_rewards_or_advantages * clipped_ratio
        )
    elif importance_reweighting_method == "gspo":
        # 这里的 group 维度是？
        # 最小计算单元：一个 rollout 回答
        # GSPO J = min(A * s, A * clip(s))
        assert old_log_probs is not None
        assert cliprange is not None
        assert response_mask is not None

        # [N, T]
        log_ratio = policy_log_probs - old_log_probs    # 对每个tok来说，是 log (pi theta / pi 0)
        mask = response_mask.to(log_ratio.dtype)
        seq_lens = mask.sum(dim=-1, keepdim=True).clamp_min(1)  # 每条 rollout 的 response 长度: [N, 1]
        # log 几何平均: [N, 1]
        mean_log_ratio = (log_ratio * mask).sum(dim=-1, keepdim=True) / seq_lens
        seq_weight = torch.exp(mean_log_ratio)   # [B, 1]，scalar for a rollout
        clipped_weight = seq_weight.clamp(1 - cliprange, 1 + cliprange)
        #   每条 rollout 一个 GSPO objective: [N, 1]
        seq_objective = torch.minimum(
                raw_rewards_or_advantages * seq_weight,
                raw_rewards_or_advantages * clipped_weight
            )
        per_token_policy_gradient_loss = -seq_objective.expand_as(policy_log_probs) # 所以loss属于整个rollout

    metadata = {
        
    }
    return per_token_policy_gradient_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence","constant"] ="sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    '''
    params: 
    per_token_policy_gradient_loss: torch.Tensor Shape (batch_size, sequence_length)
    mask                          : torch.Tensor of shape (batch_size, sequence_length) denoting which 
                                    positions should be included in the loss
    Returns:
    loss: torch.Tensor A scalar containing the average loss. Make 
        sure you can later call backward on this loss.
    '''
    masked_loss = per_token_policy_gradient_loss.where(mask, 
                                        torch.zeros_like(per_token_policy_gradient_loss))
    if loss_normalization == 'sequence':
        loss = (masked_loss.sum(dim=-1) / mask.sum(dim=-1)).mean()
    elif loss_normalization == 'constant':
        loss = masked_loss.sum().divide(normalization_constant)
    else:
      raise NotImplementedError
    return loss


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],   ### from model
    repeated_ground_truths: list[str],  ### from dataset label
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    autocast_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """ one standard on-policy GRPO optimizer update."""

    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    device = next(model.parameters()).device
    input_ids = tokenized["input_ids"].to(device)   # flattened: shape=(batch_size, max_seq_len-1)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)
    metadata = {}

    batch_size = input_ids.shape[0]
    assert batch_size % gradient_accumulation_steps == 0
    microbatch_size = batch_size // gradient_accumulation_steps

    # compute rollout rewards.
    raw_rewards, row_rewards_metadata = compute_rollout_rewards(
                reward_fn, rollout_responses, repeated_ground_truths)

    # compute standard GRPO advantages.
    assert batch_size % group_size == 0
    advantages, adv_metadata = compute_group_normalized_rewards(raw_rewards, group_size,
                            baseline, advantage_eps, advantage_normalizer)   # shape (batch_size,)
    advantages = advantages.to(device)
    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)

    microbatch_size = len(input_ids) // gradient_accumulation_steps
    batch_loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    response_token_count = torch.zeros((), device=device)
    optimizer.zero_grad()

    for i in range(0, len(input_ids), microbatch_size):
        inputs_microbatch = input_ids[i:i+microbatch_size]
        labels_microbatch = labels[i:i+microbatch_size]     # (m_batch_size, seq_len)
        response_mask_microbatch = response_mask[i:i+microbatch_size]
        advantages_microbatch = advantages[i:i+microbatch_size]
        old_log_probs_microbatch = old_log_probs[i:i+microbatch_size]

        # Forward pass.
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            logits = model(input_ids=inputs_microbatch).logits   # run the model on input_ids[start:end],
            log_probs = torch.log_softmax(logits, dim=-1)   # shape (m_batch_size, seq_len, vocab_size)
            with torch.no_grad():
                token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1) # [micro_batch, seq_len],
                entropy_sum += token_entropy.masked_select(response_mask_microbatch).sum()
                response_token_count += response_mask_microbatch.sum()
            policy_log_probs = torch.gather(log_probs, dim=-1, index=labels_microbatch.unsqueeze(-1)).squeeze(-1)   # (m_batch_size, seq_len)
            # compute per-token policy-gradient loss from advantages[start:end]

            per_token_policy_gradient_loss, per_token_policy_gradient_loss_metadata = \
                        compute_policy_gradient_loss(raw_rewards_or_advantages=advantages_microbatch,
                                         policy_log_probs=policy_log_probs,
                                         importance_reweighting_method=importance_reweighting_method,
                                         old_log_probs=old_log_probs_microbatch,
                                         cliprange=cliprange,
                                         response_mask=response_mask_microbatch)
            # aggregate over response_mask[start:end]
            loss = aggregate_loss_across_microbatch(per_token_policy_gradient_loss,
                                                    mask=response_mask_microbatch,
                                                    loss_normalization=loss_normalization,
                                                    normalization_constant=normalization_constant)
        #  divide by gradient_accumulation_steps before backward()
        loss = loss / gradient_accumulation_steps
        batch_loss = batch_loss + loss.detach()
        # Backward pass, cumulate the .grad attribute
        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()
        del logits, log_probs, token_entropy, policy_log_probs, per_token_policy_gradient_loss, loss

    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grads = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
        grad_norm = torch.stack(grads).norm(2) if grads else torch.tensor(0.0, device=device)

    # Update weights once across entire batch.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if grad_scaler is None:
        optimizer.step()
    else:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    # Zero gradients once across entire batch.
    optimizer.zero_grad()

    # Suggested metadata:
    # - reward means from compute_rollout_rewards
    # - reward/advantage stats
    # - grad_norm before clipping
    metadata.update(row_rewards_metadata)
    metadata.update(adv_metadata)
    metadata.update(per_token_policy_gradient_loss_metadata)
    metadata["grad_norm_before_clipping"] = grad_norm
    metadata["token_entropy"] = entropy_sum / response_token_count.clamp_min(1)
    metadata["response_length"] = response_token_count / batch_size
    return batch_loss, metadata
