## 4 Group Relative Policy Optimization

#### log-derivative trick

<img src="/Users/xinyue/Library/Application Support/typora-user-images/image-20260911092311050.png" alt="image-20260911092311050" style="zoom:50%;" />

#### Baselines



## 6 Off-policy

### Problem (derive_surrogate_objectives)

#### 6.1 sequence-level reweighting

解决了 bias 的问题，带来高方差

#### 6.2 PPO/GRPO-style importance reweighting and clipping

##### token-level reweighting

解决 sequence-level 的方差太大的问题，同时带来 bias

##### clipping

防止一次更新太多