GRPO Trainer



[![Hugging Face's logo](/front/assets/huggingface_logo-noborder.svg)
Hugging Face](/)

* [Models](/models)
* [Datasets](/datasets)
* [Spaces](/spaces)
* Community
* [Docs](/docs)
* [Enterprise](/enterprise)
* [Pricing](/pricing)
* ---
* [Log In](/login)
* [Sign Up](/join)

TRL documentation

GRPO Trainer

TRL
===

🏡 View all docsAWS Trainium & InferentiaAccelerateArgillaAutoTrainBitsandbytesChat UIDataset viewerDatasetsDeploying on AWSDiffusersDistilabelEvaluateGradioHubHub Python LibraryHuggingface.jsInference Endpoints (dedicated)Inference ProvidersLeRobotLeaderboardsLightevalMicrosoft AzureOptimumPEFTSafetensorsSentence TransformersTRLTasksText Embeddings InferenceText Generation InferenceTokenizersTransformersTransformers.jssmolagentstimm

Search documentation

mainv0.20.0v0.19.1v0.18.1v0.17.0v0.16.1v0.15.2v0.14.0v0.13.0v0.12.2v0.11.4v0.10.1v0.9.6v0.8.6v0.7.11v0.6.0v0.5.0v0.4.7v0.3.1v0.2.1v0.1.1
EN

Getting started

[TRL](/docs/trl/main/en/index)[Installation](/docs/trl/main/en/installation)[Quickstart](/docs/trl/main/en/quickstart)

Conceptual Guides

[Dataset Formats](/docs/trl/main/en/dataset_formats)[Paper Index](/docs/trl/main/en/paper_index)[Training FAQ](/docs/trl/main/en/how_to_train)[Understanding Logs](/docs/trl/main/en/logging)

How-to guides

[Command Line Interface (CLI)](/docs/trl/main/en/clis)[Customizing the Training](/docs/trl/main/en/customization)[Reducing Memory Usage](/docs/trl/main/en/reducing_memory_usage)[Speeding Up Training](/docs/trl/main/en/speeding_up_training)[Distributing Training](/docs/trl/main/en/distributing_training)[Using Trained Models](/docs/trl/main/en/use_model)

Integrations

[DeepSpeed](/docs/trl/main/en/deepspeed_integration)[Liger Kernel](/docs/trl/main/en/liger_kernel_integration)[PEFT](/docs/trl/main/en/peft_integration)[Unsloth](/docs/trl/main/en/unsloth_integration)[vLLM](/docs/trl/main/en/vllm_integration)

Examples

[Example Overview](/docs/trl/main/en/example_overview)[Community Tutorials](/docs/trl/main/en/community_tutorials)[Sentiment Tuning](/docs/trl/main/en/sentiment_tuning)[Training StackLlama](/docs/trl/main/en/using_llama_models)[Detoxifying a Language Model](/docs/trl/main/en/detoxifying_a_lm)[Multi Adapter RLHF](/docs/trl/main/en/multi_adapter_rl)[Fine-tuning a Multimodal Model Using SFT (Single or Multi-Image Dataset)](/docs/trl/main/en/training_vlm_sft)

API

Trainers

[AlignProp](/docs/trl/main/en/alignprop_trainer)[BCO](/docs/trl/main/en/bco_trainer)[CPO](/docs/trl/main/en/cpo_trainer)[DDPO](/docs/trl/main/en/ddpo_trainer)[DPO](/docs/trl/main/en/dpo_trainer)[Online DPO](/docs/trl/main/en/online_dpo_trainer)[GKD](/docs/trl/main/en/gkd_trainer)[GRPO](/docs/trl/main/en/grpo_trainer)[KTO](/docs/trl/main/en/kto_trainer)[Nash-MD](/docs/trl/main/en/nash_md_trainer)[ORPO](/docs/trl/main/en/orpo_trainer)[PPO](/docs/trl/main/en/ppo_trainer)[PRM](/docs/trl/main/en/prm_trainer)[Reward](/docs/trl/main/en/reward_trainer)[RLOO](/docs/trl/main/en/rloo_trainer)[SFT](/docs/trl/main/en/sft_trainer)[Iterative SFT](/docs/trl/main/en/iterative_sft_trainer)[XPO](/docs/trl/main/en/xpo_trainer)

[Model Classes](/docs/trl/main/en/models)[Model Utilities](/docs/trl/main/en/model_utils)[Best of N Sampling](/docs/trl/main/en/best_of_n)[Judges](/docs/trl/main/en/judges)[Callbacks](/docs/trl/main/en/callbacks)[Data Utilities](/docs/trl/main/en/data_utils)[Reward Functions](/docs/trl/main/en/rewards)[Script Utilities](/docs/trl/main/en/script_utils)[Others](/docs/trl/main/en/others)

You are viewing main version, which requires [installation from source](/docs/trl/installation#source). If you'd like
regular pip install, checkout the latest stable version ([v0.20.0](/docs/trl/v0.20.0/grpo_trainer)).

![Hugging Face's logo](/front/assets/huggingface_logo-noborder.svg)

Join the Hugging Face community

and get access to the augmented documentation experience

Collaborate on models, datasets and Spaces

Faster examples with accelerated inference

Switch between documentation themes

[Sign Up](/join)

to get started

GRPO Trainer
============

[![](https://img.shields.io/badge/All_models-GRPO-blue)](https://huggingface.co/models?other=grpo,trl)

Overview
--------

TRL supports the GRPO Trainer for training language models, as described in the paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300) by [Zhihong Shao](https://huggingface.co/syhia), [Peiyi Wang](https://huggingface.co/peiyiwang89), [Qihao Zhu](https://huggingface.co/zqh11), Runxin Xu, [Junxiao Song](https://huggingface.co/haha-point), Mingchuan Zhang, Y. K. Li, Y. Wu, [Daya Guo](https://huggingface.co/guoday).

The abstract from the paper is the following:

> Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.

This post-training method was contributed by [Quentin Gallouédec](https://huggingface.co/qgallouedec).

Quick start
-----------

This example demonstrates how to train a model using the GRPO method. We train a [Qwen 0.5B Instruct model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) with the prompts from the [TLDR dataset](https://huggingface.co/datasets/trl-lib/tldr) (completion column is ignored!). You can view the data in the dataset here:

Below is the script to train the model.

Copied

```
# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO")
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

Execute the script using the following command:

Copied

```
accelerate launch train_grpo.py
```

Distributed across 8 GPUs, the training takes approximately 1 day.

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/grpo_curves.png)

Looking deeper into the GRPO method
-----------------------------------

GRPO is an online learning algorithm, meaning it improves iteratively by using the data generated by the trained model itself during training. The intuition behind GRPO objective is to maximize the advantage of the generated completions, while ensuring that the model remains close to the reference policy. To understand how GRPO works, it can be broken down into four main steps: **Generating completions**, **computing the advantage**, **estimating the KL divergence**, and **computing the loss**.

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/grpo_visual.png)

### Generating completions

At each training step, we sample a batch of prompts and generate a set of G G G completions for each prompt (denoted as oi o\_i oi​).

### Computing the advantage

For each of the G G G sequences, we compute the reward using a reward model. To align with the comparative nature of reward models—typically trained on datasets of comparisons between outputs for the same question—the advantage is calculated to reflect these relative comparisons. It is normalized as follows:
A^i,t=ri−mean(r)std(r)\hat{A}\_{i,t} = \frac{r\_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}A^i,t​=std(r)ri​−mean(r)​

This approach gives the method its name: **Group Relative Policy Optimization (GRPO)**.

It was shown in the paper [Understanding R1-Zero-Like Training: A Critical Perspective](https://huggingface.co/papers/2503.20783) that scaling by std(r) \text{std}(\mathbf{r}) std(r) may cause a question-level difficulty bias. You can disable this scaling by setting `scale_rewards=False` in [GRPOConfig](/docs/trl/main/en/grpo_trainer#trl.GRPOConfig).

### Estimating the KL divergence

KL divergence is estimated using the approximator introduced by [Schulman et al. (2020)](http://joschu.net/blog/kl-approx.html). The approximator is defined as follows:
DKL[πθ∥πref]=πref(oi,t∣q,oi,<t)πθ(oi,t∣q,oi,<t)−log⁡πref(oi,t∣q,oi,<t)πθ(oi,t∣q,oi,<t)−1,\mathbb{D}\_{\text{KL}}\left[\pi\_\theta \|\pi\_{\text{ref}}\right] = \frac{\pi\_{\text{ref}}(o\_{i,t} \mid q, o\_{i,<t})}{\pi\_\theta(o\_{i,t} \mid q, o\_{i,<t})} - \log \frac{\pi\_{\text{ref}}(o\_{i,t} \mid q, o\_{i,<t})}{\pi\_\theta(o\_{i,t} \mid q, o\_{i,<t})} - 1,
DKL​[πθ​∥πref​]=πθ​(oi,t​∣q,oi,<t​)πref​(oi,t​∣q,oi,<t​)​−logπθ​(oi,t​∣q,oi,<t​)πref​(oi,t​∣q,oi,<t​)​−1,

### Computing the loss

The objective is to maximize the advantage while ensuring that the model remains close to the reference policy. Consequently, the loss is defined as follows:
LGRPO(θ)=−1∑i=1G∣oi∣∑i=1G∑t=1∣oi∣[πθ(oi,t∣q,oi,<t)[πθ(oi,t∣q,oi,<t)]no gradA^i,t−βDKL[πθ∥πref]],
\mathcal{L}\_{\text{GRPO}}(\theta) = -\frac{1}{\sum\_{i=1}^G |o\_i|} \sum\_{i=1}^G \sum\_{t=1}^{|o\_i|} \left[ \frac{\pi\_\theta(o\_{i,t} \mid q, o\_{i,< t})}{\left[\pi\_\theta(o\_{i,t} \mid q, o\_{i,< t})\right]\_{\text{no grad}}} \hat{A}\_{i,t} - \beta \mathbb{D}\_{\text{KL}}\left[\pi\_\theta \| \pi\_{\text{ref}}\right] \right],
LGRPO​(θ)=−∑i=1G​∣oi​∣1​i=1∑G​t=1∑∣oi​∣​[[πθ​(oi,t​∣q,oi,<t​)]no grad​πθ​(oi,t​∣q,oi,<t​)​A^i,t​−βDKL​[πθ​∥πref​]],

where the first term represents the scaled advantage and the second term penalizes deviations from the reference policy through KL divergence.

Note that compared to the original formulation in [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300), we don’t scale by 1∣oi∣ \frac{1}{|o\_i|} ∣oi​∣1​ because it was shown in the paper [Understanding R1-Zero-Like Training: A Critical Perspective](https://huggingface.co/papers/2503.20783) that this introduces a response-level length bias. More details in [loss types](#loss-types).

Note that compared to the original formulation in [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300), we use β=0.0 \beta = 0.0 β=0.0 by default, meaning that the KL divergence term is not used. This choice is motivated by several recent studies (e.g., [Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model](https://huggingface.co/papers/2503.24290)) which have shown that the KL divergence term is not essential for training with GRPO. As a result, it has become common practice to exclude it (e.g. [Understanding R1-Zero-Like Training: A Critical Perspective](https://huggingface.co/papers/2503.20783), [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://huggingface.co/papers/2503.14476)). If you wish to include the KL divergence term, you can set `beta` in [GRPOConfig](/docs/trl/main/en/grpo_trainer#trl.GRPOConfig) to a non-zero value.

In the original paper, this formulation is generalized to account for multiple updates after each generation (denoted μ \mu μ, can be set with `num_iterations` in [GRPOConfig](/docs/trl/main/en/grpo_trainer#trl.GRPOConfig)) by leveraging the **clipped surrogate objective**:
LGRPO(θ)=−1∑i=1G∣oi∣∑i=1G∑t=1∣oi∣[min⁡(πθ(oi,t∣q,oi,<t)πθold(oi,t∣q,oi,<t)A^i,t, clip(πθ(oi,t∣q,oi,<t)πθold(oi,t∣q,oi,<t),1−ϵ,1+ϵ)A^i,t)−βDKL[πθ∥πref]],
\mathcal{L}\_{\text{GRPO}}(\theta) = - \frac{1}{\sum\_{i=1}^G |o\_i|} \sum\_{i=1}^G \sum\_{t=1}^{|o\_i|} \left[ \min \left( \frac{\pi\_\theta(o\_{i,t} \mid q, o\_{i,< t})}{\pi\_{\theta\_{\text{old}}}(o\_{i,t} \mid q, o\_{i,< t})} \hat{A}\_{i,t}, \, \text{clip}\left( \frac{\pi\_\theta(o\_{i,t} \mid q, o\_{i,< t})}{\pi\_{\theta\_{\text{old}}}(o\_{i,t} \mid q, o\_{i,< t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}\_{i,t} \right) - \beta \mathbb{D}\_{\text{KL}}\left[\pi\_\theta \| \pi\_{\text{ref}}\right] \right],
LGRPO​(θ)=−∑i=1G​∣oi​∣1​i=1∑G​t=1∑∣oi​∣​[min(πθold​​(oi,t​∣q,oi,<t​)πθ​(oi,t​∣q,oi,<t​)​A^i,t​,clip(πθold​​(oi,t​∣q,oi,<t​)πθ​(oi,t​∣q,oi,<t​)​,1−ϵ,1+ϵ)A^i,t​)−βDKL​[πθ​∥πref​]],

where clip(⋅,1−ϵ,1+ϵ)\text{clip}(\cdot, 1 - \epsilon, 1 + \epsilon) clip(⋅,1−ϵ,1+ϵ) ensures that updates do not deviate excessively from the reference policy by bounding the policy ratio between 1−ϵ 1 - \epsilon 1−ϵ and 1+ϵ 1 + \epsilon 1+ϵ.
When μ=1 \mu = 1 μ=1 (default in TRL), the clipped surrogate objective simplifies to the original objective.

#### Loss Types

Several formulations of the objective have been proposed in the literature. Initially, the objective of GRPO was defined as follows:
LGRPO(θ)=−1G∑i=1G1∣oi∣∑t=1∣oi∣li,t,
\mathcal{L}\_{\text{GRPO}}(\theta) = - \frac{1}{G} \sum\_{i=1}^G \frac{1}{|o\_i|} \sum\_{t=1}^{|o\_i|} l\_{i,t},
LGRPO​(θ)=−G1​i=1∑G​∣oi​∣1​t=1∑∣oi​∣​li,t​,

where
li,t=πθ(oi,t∣q,oi,<t)[πθ(oi,t∣q,oi,<t)]no gradA^i,t−βDKL[πθ∥πref].
l\_{i,t} = \frac{\pi\_\theta(o\_{i,t} \mid q, o\_{i,< t})}{\left[\pi\_\theta(o\_{i,t} \mid q, o\_{i,< t})\right]\_{\text{no grad}}} \hat{A}\_{i,t} - \beta \mathbb{D}\_{\text{KL}}\left[\pi\_\theta \| \pi\_{\text{ref}}\right].
li,t​=[πθ​(oi,t​∣q,oi,<t​)]no grad​πθ​(oi,t​∣q,oi,<t​)​A^i,t​−βDKL​[πθ​∥πref​].

The [DAPO paper](https://huggingface.co/papers/2503.14476) highlights the limitations of the GRPO algorithm’s sample-level loss in long-CoT scenarios, where longer responses are under-penalized, leading to poorer quality outputs. The proposed solution is a token-level normalization, which better handles longer sequences by assigning more balanced rewards to individual tokens, regardless of response length:
LDAPO(θ)=−1∑i=1G∣oi∣∑i=1G∑t=1∣oi∣li,t,
\mathcal{L}\_{\text{DAPO}}(\theta) = - \frac{1}{\sum\_{i=1}^G |o\_i|} \sum\_{i=1}^G \sum\_{t=1}^{|o\_i|} l\_{i,t},
LDAPO​(θ)=−∑i=1G​∣oi​∣1​i=1∑G​t=1∑∣oi​∣​li,t​,

Furthermore, it was demonstrated in the paper [Understanding R1-Zero-Like Training: A Critical Perspective](https://huggingface.co/papers/2503.20783) that the initial GRPO formulation introduces a response length bias. They show that while the DAPO formulation reduces this bias, it does not eliminate it completely. To fully remove this bias, they propose dividing by a constant instead of the sequence length, resulting in the following formulation:
LDr. GRPO(θ)=−1LG∑i=1G∑t=1∣oi∣li,t,
\mathcal{L}\_{\text{Dr. GRPO}}(\theta) = - \frac{1}{LG} \sum\_{i=1}^G \sum\_{t=1}^{|o\_i|} l\_{i,t},
LDr. GRPO​(θ)=−LG1​i=1∑G​t=1∑∣oi​∣​li,t​,

This constant is recommended to be the maximum completion length. To use this formulation, set `loss_type="dr_grpo"` in the [GRPOConfig](/docs/trl/main/en/grpo_trainer#trl.GRPOConfig).

Logged metrics
--------------

* `num_tokens`: The total number of tokens processed so far, including both prompts and completions.
* `completions/mean_length`: The average length of generated completions.
* `completions/min_length`: The minimum length of generated completions.
* `completions/max_length`: The maximum length of generated completions.
* `completions/mean_terminated_length`: The average length of generated completions that terminate with EOS.
* `completions/min_terminated_length`: The minimum length of generated completions that terminate with EOS.
* `completions/max_terminated_length`: The maximum length of generated completions that terminate with EOS.
* `completions/clipped_ratio` : The ratio of truncated (clipped) completions.
* `reward/{reward_func_name}/mean`: The average reward from a specific reward function.
* `reward/{reward_func_name}/std`: The standard deviation of the reward from a specific reward function.
* `reward`: The overall average reward after applying reward weights.
* `reward_std`: The standard deviation of the overall reward within each batch after applying reward weights.
* `frac_reward_zero_std`: The fraction of samples in the generation batch with a reward std of zero, implying there is little diversity for that prompt (all answers are correct or incorrect).
* `entropy`: Average entropy of token predictions across generated completions. (If `mask_truncated_completions=True`, masked sequences tokens are excluded.)
* `kl`: The average KL divergence between the model and the reference model, calculated over generated completions. Logged only if `beta` is nonzero.
* `clip_ratio/region_mean`: The ratio of token (or sequence, if `importance_sampling_level="sequence"`) probabilities where the GRPO objective is clipped to stay within the trust region:clip(ri,t(θ),1−ϵlow,1+ϵhigh) ,ri,t(θ)=πθ(oi,t∣q,oi,<t)πθold(oi,t∣q,oi,<t) .
  \text{clip}\left( r\_{i,t}(\theta), 1 - \epsilon\_\mathrm{low}, 1 + \epsilon\_\mathrm{high} \right)\,, \qquad r\_{i,t}(\theta) = \frac{\pi\_\theta(o\_{i,t} \mid q, o\_{i,< t})}{\pi\_{\theta\_{\text{old}}}(o\_{i,t} \mid q, o\_{i,< t})}\,.
  clip(ri,t​(θ),1−ϵlow​,1+ϵhigh​),ri,t​(θ)=πθold​​(oi,t​∣q,oi,<t​)πθ​(oi,t​∣q,oi,<t​)​.
  A higher value means more tokens are clipped, which constrains how much the policy $\pi\_\theta$ can change.
* `clip_ratio/low_mean`: The average ratio of token (or sequence, if `importance_sampling_level="sequence"`) probabilities that were clipped on the lower bound of the trust region: ri,t(θ)<1−ϵlowr\_{i,t}(\theta) < 1 - \epsilon\_\mathrm{low}ri,t​(θ)<1−ϵlow​
* `clip_ratio/low_min`: The minimum ratio of token (or sequence, if `importance_sampling_level="sequence"`) probabilities that were clipped on the lower bound of the trust region: ri,t(θ)<1−ϵlowr\_{i,t}(\theta) < 1 - \epsilon\_\mathrm{low}ri,t​(θ)<1−ϵlow​
* `clip_ratio/high_mean`: The average ratio of token (or sequence, if `importance_sampling_level="sequence"`) probabilities that were clipped on the upper bound of the trust region: ri,t(θ)>1+ϵhighr\_{i,t}(\theta) > 1 + \epsilon\_\mathrm{high}ri,t​(θ)>1+ϵhigh​
* `clip_ratio/high_max`: The maximum ratio of token (or sequence, if `importance_sampling_level="sequence"`) probabilities that were clipped on the upper bound of the trust region: ri,t(θ)>1+ϵhighr\_{i,t}(\theta) > 1 + \epsilon\_\mathrm{high}ri,t​(θ)>1+ϵhigh​.

Customization
-------------

### Speed up training with vLLM-powered generation

Generation is often the main bottleneck when training with online methods. To accelerate generation, you can use [vLLM](https://github.com/vllm-project/vllm), a high-throughput, low-latency inference engine for LLMs. To enable it, first install the package with

Copied

```
pip install trl[vllm]
```

We support two ways of using vLLM during training: **server mode** and **colocate mode**.

#### 🔌 Option 1: Server mode

In this mode, vLLM runs in a separate process (and using separate GPUs) and communicates with the trainer via HTTP. This is ideal if you have dedicated GPUs for inference.

1. **Start the vLLM server**:

   Copied

   ```
   trl vllm-serve --model <model_name>
   ```
2. **Enable server mode in your training script**:

   Copied

   ```
   from trl import GRPOConfig

   training_args = GRPOConfig(
       ...,
       use_vllm=True,
       vllm_mode="server",  # default value, can be omitted
   )
   ```

Make sure that the server is using different GPUs than the trainer, otherwise you may run into NCCL errors. You can specify the GPUs to use with the `CUDA_VISIBLE_DEVICES` environment variable.

#### 🧩 Option 2: Colocate mode

In this mode, vLLM runs inside the trainer process and shares GPU memory with the training model. This avoids launching a separate server and can improve GPU utilization, but may lead to memory contention on the training GPUs.

Copied

```
from trl import GRPOConfig

training_args = GRPOConfig(
    ...,
    use_vllm=True,
    vllm_mode="colocate",
)
```

Depending on the model size and the overall GPU memory requirements for training, you may need to adjust the `vllm_gpu_memory_utilization` parameter in [GRPOConfig](/docs/trl/main/en/grpo_trainer#trl.GRPOConfig) to avoid underutilization or out-of-memory errors.

We provide a [HF Space](https://huggingface.co/spaces/trl-lib/recommend-vllm-memory) to help estimate the recommended GPU memory utilization based on your model configuration and experiment settings. Simply use it as follows to get `vllm_gpu_memory_utilization` recommendation:

If the recommended value does not work in your environment, we suggest adding a small buffer (e.g., +0.05 or +0.1) to the recommended value to ensure stability.

By default, GRPO uses `MASTER_ADDR=localhost` and `MASTER_PORT=12345` for vLLM, but you can override these values by setting the environment variables accordingly.

For more information, see [Speeding up training with vLLM](speeding_up_training#vllm-for-fast-generation-in-online-methods).

### GRPO at scale: train a 70B+ Model on multiple nodes

When training large models like **Qwen2.5-72B**, you need several key optimizations to make the training efficient and scalable across multiple GPUs and nodes. These include:

* **DeepSpeed ZeRO Stage 3**: ZeRO leverages data parallelism to distribute model states (weights, gradients, optimizer states) across multiple GPUs and CPUs, reducing memory and compute requirements on each device. Since large models cannot fit on a single GPU, using ZeRO Stage 3 is required for training such model. For more details, see [DeepSpeed Integration](deepspeed_integration).
* **Accelerate**: Accelerate is a library that simplifies distributed training across multiple GPUs and nodes. It provides a simple API to launch distributed training and handles the complexities of distributed training, such as data parallelism, gradient accumulation, and distributed data loading. For more details, see [Distributing Training](distributing_training).
* **vLLM**: See the previous section on how to use vLLM to speed up generation.

Below is an example SLURM script to train a 70B model with GRPO on multiple nodes. This script trains a model on 4 nodes and uses the 5th node for vLLM-powered generation.

Copied

```
#!/bin/bash
#SBATCH --nodes=5
#SBATCH --gres=gpu:8

# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Assign the first 4 nodes for training and the 5th node for vLLM
TRAIN_NODES="${NODELIST[@]:0:4}"  # Nodes 0, 1, 2, 3 for training
VLLM_NODE="${NODELIST[4]}"  # Node 4 for vLLM

# Run training on the first 4 nodes (Group 1)
srun --nodes=4 --ntasks=4 --nodelist="${NODELIST[@]:0:4}" accelerate launch \
     --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
     --num_processes 32 \
     --num_machines 4 \
     --main_process_ip ${NODELIST[0]} \
     --machine_rank $SLURM_PROCID \
     --rdzv_backend c10d \
     train_grpo.py \
     --server_ip $VLLM_NODE &

# Run vLLM server on the 5th node (Group 2)
srun --nodes=1 --ntasks=1 --nodelist="${NODELIST[4]}" trl vllm-serve --model Qwen/Qwen2.5-72B --tensor_parallel_size 8 &

wait
```

Copied

```
import argparse

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm_server_host", type=str, default="", help="The server IP")
    args = parser.parse_args()

    # Example dataset from TLDR
    dataset = load_dataset("trl-lib/tldr", split="train")

    # Dummy reward function: count the number of unique characters in the completions
    def reward_num_unique_chars(completions, **kwargs):
        return [len(set(c)) for c in completions]

    training_args = GRPOConfig(
        output_dir="Qwen2.5-72B-GRPO",
        per_device_train_batch_size=4,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_server_host=args.vllm_server_host.replace("ip-", "").replace("-", "."),  # from ip-X-X-X-X to X.X.X.X
    )

    trainer = GRPOTrainer(model="Qwen/Qwen2.5-72B", args=training_args, reward_funcs=reward_num_unique_chars, train_dataset=dataset)
    trainer.train()

if __name__=="__main__":
    main()
```

### Using a custom reward function

The [GRPOTrainer](/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer) supports using custom reward functions instead of dense reward models. To ensure compatibility, your reward function must satisfy the following requirements:

1. **Input arguments**:

   * The function must accept the following as keyword arguments:

     + `prompts` (contains the prompts),
     + `completions` (contains the generated completions),
     + `completions_ids` (contains the tokenized completions),
     + `trainer_state` ([TrainerState](https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.TrainerState)): The current state of the trainer. This can be used to implement dynamic reward functions, such as curriculum learning, where the reward is adjusted based on the training progress.
     + All columns names (but `prompt`) that the dataset may have. For example, if the dataset contains a column named `ground_truth`, the function will be called with `ground_truth` as a keyword argument.

     The easiest way to comply with this requirement is to use `**kwargs` in the function signature.
   * Depending on the dataset format, the input will vary:

     + For [standard format](dataset_formats#standard), `prompts` and `completions` will be lists of strings.
     + For [conversational format](dataset_formats#conversational), `prompts` and `completions` will be lists of message dictionaries.
2. **Return value**: The function must return a list of floats. Each float represents the reward corresponding to a single completion.

#### Example 1: Reward longer completions

Below is an example of a reward function for a standard format that rewards longer completions:

Copied

```
def reward_func(completions_ids, **kwargs):
    """Reward function that assigns higher scores to longer completions (in terms of token count)."""
    return [float(len(ids)) for ids in completions_ids]
```

You can test it as follows:

Copied

```
>>> prompts = ["The sky is", "The sun is"]  # not used in the reward function, but the trainer will pass it
>>> completions = [" blue.", " in the sky."]  # not used in the reward function, but the trainer will pass it
>>> completions_ids = [[6303, 13], [304, 279, 12884, 13]]
>>> reward_func(prompts=prompts, completions=completions, completions_ids=completions_ids)
[2.0, 4.0]
```

#### Example 1.1: Reward longer completions (based in the number of characters)

Same as the previous example, but this time the reward function is based on the number of characters instead of tokens.

Copied

```
def reward_func(completions, **kwargs):
    """Reward function that assigns higher scores to longer completions (in terms of character count)."""
    return [float(len(completion)) for completion in completions]
```

You can test it as follows:

Copied

```
>>> prompts = ["The sky is", "The sun is"]
>>> completions = [" blue.", " in the sky."]
>>> completions_ids = [[6303, 13], [304, 279, 12884, 13]]  # not used in the reward function, but the trainer will pass it
>>> reward_func(prompts=prompts, completions=completions, completions_ids=completions_ids)
[6.0, 12.0]
```

#### Example 2: Reward completions with specific format

Below is an example of a reward function that checks if the completion has a specific format. This example is inspired by the *format reward* function used in the paper [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://huggingface.co/papers/2501.12948).
It is designed for conversational format, where prompts and completions consist of structured messages.

Copied

```
import re

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
```

You can test this function as follows:

Copied

```
>>> prompts = [
...     [{"role": "assistant", "content": "What is the result of (1 + 2) * 4?"}],
...     [{"role": "assistant", "content": "What is the result of (3 + 1) * 2?"}],
... ]
>>> completions = [
...     [{"role": "assistant", "content": "<think>The sum of 1 and 2 is 3, which we multiply by 4 to get 12.</think><answer>(1 + 2) * 4 = 12</answer>"}],
...     [{"role": "assistant", "content": "The sum of 3 and 1 is 4, which we multiply by 2 to get 8. So (3 + 1) * 2 = 8."}],
... ]
>>> format_reward_func(prompts=prompts, completions=completions)
[1.0, 0.0]
```

#### Example 3: Reward completions based on a reference

Below is an example of a reward function that checks if the completion is correct. This example is inspired by the *accuracy reward* function used in the paper [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://huggingface.co/papers/2501.12948).
This example is designed for [standard format](dataset_formats#standard), where the dataset contains a column named `ground_truth`.

Copied

```
import re

def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
```

You can test this function as follows:

Copied

```
>>> prompts = ["Problem: Solve the equation $2x + 3 = 7$. Solution:", "Problem: Solve the equation $3x - 5 = 10$."]
>>> completions = [r" The solution is \boxed{2}.", r" The solution is \boxed{6}."]
>>> ground_truth = ["2", "5"]
>>> reward_func(prompts=prompts, completions=completions, ground_truth=ground_truth)
[1.0, 0.0]
```

#### Example 4: Multi-task reward functions

Below is an example of using multiple reward functions in the [GRPOTrainer](/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer). In this example, we define two task-specific reward functions: `math_reward_func` and `coding_reward_func`. The `math_reward_func` rewards math problems based on their correctness, while the `coding_reward_func` rewards coding problems based on whether the solution works.

Copied

```
from datasets import Dataset
from trl import GRPOTrainer

# Define a dataset that contains both math and coding problems
dataset = Dataset.from_list(
    [
        {"prompt": "What is 2+2?", "task": "math"},
        {"prompt": "Write a function that returns the sum of two numbers.", "task": "code"},
        {"prompt": "What is 3*4?", "task": "math"},
        {"prompt": "Write a function that returns the product of two numbers.", "task": "code"},
    ]
)

# Math-specific reward function
def math_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "math":
            # Calculate math-specific reward
            correct = check_math_solution(prompt, completion)
            reward = 1.0 if correct else -1.0
            rewards.append(reward)
        else:
            # Return None for non-math tasks
            rewards.append(None)
    return rewards

# Coding-specific reward function
def coding_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "coding":
            # Calculate coding-specific reward
            works = test_code_solution(prompt, completion)
            reward = 1.0 if works else -1.0
            rewards.append(reward)
        else:
            # Return None for non-coding tasks
            rewards.append(None)
    return rewards

# Use both task-specific reward functions
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=[math_reward_func, coding_reward_func],
    train_dataset=dataset,
)

trainer.train()
```

In this example, the `math_reward_func` and `coding_reward_func` are designed to work with a mixed dataset that contains both math and coding problems. The `task` column in the dataset is used to determine which reward function to apply to each problem. If there is no relevant reward function for a sample in the dataset, the reward function will return `None` and the [GRPOTrainer](/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer) will continue with the valid functions and tasks. This allows the [GRPOTrainer](/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer) to handle multiple reward functions with different applicability.

Note that the [GRPOTrainer](/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer) will ignore the `None` rewards returned by the reward functions and only consider the rewards returned by the relevant functions. This ensures that the model is trained on the relevant tasks and ignores the tasks for which there is no relevant reward function.

#### Passing the reward function to the trainer

To use your custom reward function, pass it to the [GRPOTrainer](/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer) as follows:

Copied

```
from trl import GRPOTrainer

trainer = GRPOTrainer(
    reward_funcs=reward_func,
    ...,
)
```

If you have multiple reward functions, you can pass them as a list:

Copied

```
from trl import GRPOTrainer

trainer = GRPOTrainer(
    reward_funcs=[reward_func1, reward_func2],
    ...,
)
```

and the reward will be computed as the sum of the rewards from each function, or the weighted sum if `reward_weights` is provided in the config.

Note that [GRPOTrainer](/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer) supports multiple reward functions of different types. See the parameters documentation for more details.

Vision-Language Model (VLM) Training
------------------------------------

GRPO supports training Vision-Language Models (VLMs) on multimodal datasets containing both text and images.

### Supported Models

Tested with:

* **Gemma3** — e.g., `google/gemma-3-4b-it`
* **LLaVA-NeXT** — e.g., `llava-hf/llava-v1.6-mistral-7b-hf`
* **Qwen2-VL** — e.g., `Qwen/Qwen2-VL-2B-Instruct`
* **Qwen2.5-VL** — e.g., `Qwen/Qwen2.5-VL-3B-Instruct`
* **SmolVLM2** — e.g., `HuggingFaceTB/SmolVLM2-2.2B-Instruct`

Compatibility with all VLMs is not guaranteed. If you believe a model should be supported, feel free to open an issue on GitHub — or better yet, submit a pull request with the required changes.

### Quick Start

Use [grpo\_vlm.py](https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py) to fine-tune a VLM. Example command for training on [`lmms-lab/multimodal-open-r1-8k-verified`](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified):

Copied

```
accelerate launch \
  --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
  examples/scripts/grpo_vlm.py \
  --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir grpo-Qwen2.5-VL-3B-Instruct \
  --learning_rate 1e-5 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --max_prompt_length 2048 \
  --max_completion_length 1024 \
  --use_vllm \
  --vllm_mode colocate \
  --use_peft \
  --lora_target_modules "q_proj", "v_proj" \
  --log_completions
```

### Configuration Tips

VLM training may fail if image tokens are truncated. We highly recommend to disable truncation by setting `max\_prompt\_length` to `None`.

* Use LoRA on vision-language projection layers
* Enable 4-bit quantization to reduce memory usage
* VLMs are memory-intensive — start with smaller batch sizes
* Most models are compatible with vLLM (`server` and `colocate` modes)

### Dataset Format

Each training sample should include:

* `prompt`: Text formatted via the processor’s chat template
* `image`: A single image (PIL or NumPy array)

The trainer automatically handles image-to-tensor conversion via the model’s image processor.

GRPOTrainer
-----------

### class trl.GRPOTrainer

  [< source >](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py#L418)

( model: typing.Union[str, transformers.modeling\_utils.PreTrainedModel] reward\_funcs: typing.Union[str, transformers.modeling\_utils.PreTrainedModel, typing.Callable[[list, list], list[float]], list[typing.Union[str, transformers.modeling\_utils.PreTrainedModel, typing.Callable[[list, list], list[float]]]]] args: typing.Optional[trl.trainer.grpo\_config.GRPOConfig] = None train\_dataset: typing.Union[datasets.arrow\_dataset.Dataset, datasets.iterable\_dataset.IterableDataset, NoneType] = None eval\_dataset: typing.Union[datasets.arrow\_dataset.Dataset, datasets.iterable\_dataset.IterableDataset, dict[str, typing.Union[datasets.arrow\_dataset.Dataset, datasets.iterable\_dataset.IterableDataset]], NoneType] = None processing\_class: typing.Union[transformers.tokenization\_utils\_base.PreTrainedTokenizerBase, transformers.processing\_utils.ProcessorMixin, NoneType] = None reward\_processing\_classes: typing.Union[transformers.tokenization\_utils\_base.PreTrainedTokenizerBase, list[transformers.tokenization\_utils\_base.PreTrainedTokenizerBase], NoneType] = None callbacks: typing.Optional[list[transformers.trainer\_callback.TrainerCallback]] = None optimizers: tuple = (None, None) peft\_config: typing.Optional[ForwardRef('PeftConfig')] = None  )

Parameters

* **model** (`Union[str, PreTrainedModel]`) —
  Model to be trained. Can be either:
  + A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
    path to a *directory* containing model weights saved using
    [save\_pretrained](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `'./my_model_directory/'`. The model is loaded
    using [from\_pretrained](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) with the keyword arguments in
    `args.model_init_kwargs`.
  + A [PreTrainedModel](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel) object. Only causal language models are supported.
* **reward\_funcs** (`Union[RewardFunc, list[RewardFunc]]`) —
  Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
  functions with the prompts and completions and sum the rewards. Can be either:
  + A single reward function, such as:

    - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
      path to a *directory* containing model weights saved using
      [save\_pretrained](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained), e.g., `'./my_model_directory/'`. The model is loaded
      using [from\_pretrained](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSequenceClassification.from_pretrained) with `num_labels=1` and the
      keyword arguments in `args.model_init_kwargs`.
    - A [PreTrainedModel](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel) object: Only sequence classification models are supported.
    - A custom reward function: The function is provided with the prompts and the generated completions,
      plus any additional columns in the dataset. It should return a list of rewards. Custom reward
      functions can also return `None` when the reward is not applicable to those samples. This is useful
      for multi-task training where different reward functions apply to different types of samples. When a
      reward function returns `None` for a sample, that reward function is excluded from the reward
      calculation for that sample. For more details, see [Using a custom reward
      function](#using-a-custom-reward-function).

      The trainer’s state is also passed to the reward function. The trainer’s state is an instance of
      [TrainerState](https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.TrainerState) and can be accessed by accessing the `trainer_state` argument to the
      reward function’s signature.
  + A list of reward functions, where each item can independently be any of the above types. Mixing different
    types within the list (e.g., a string model ID and a custom reward function) is allowed.
* **args** ([GRPOConfig](/docs/trl/main/en/grpo_trainer#trl.GRPOConfig), *optional*, defaults to `None`) —
  Configuration for this trainer. If `None`, a default configuration is used.
* **train\_dataset** ([Dataset](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset) or [IterableDataset](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.IterableDataset)) —
  Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
  ignored. The format of the samples can be either:
  + [Standard](dataset_formats#standard): Each sample contains plain text.
  + [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
    and content).
* **eval\_dataset** ([Dataset](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset), [IterableDataset](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.IterableDataset) or `dict[str, Union[Dataset, IterableDataset]]`) —
  Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
* **processing\_class** ([PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase) or [ProcessorMixin](https://huggingface.co/docs/transformers/main/en/main_classes/processors#transformers.ProcessorMixin), *optional*, defaults to `None`) —
  Processing class used to process the data. The padding side must be set to “left”. If `None`, the
  processing class is loaded from the model’s name with [from\_pretrained](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoProcessor.from_pretrained). A
  padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
  `tokenizer.eos_token` will be used as the default.
* **reward\_processing\_classes** (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`) —
  Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:
  + A single processing class: Used when `reward_funcs` contains only one reward function.
  + A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
    If set to `None`, or if an element of the list corresponding to a [PreTrainedModel](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel) is
    `None`, the tokenizer for the model is automatically loaded using
    [from\_pretrained](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained). For elements in `reward_funcs` that are custom reward
    functions (not [PreTrainedModel](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel)), the corresponding entries in `reward_processing_classes`
    are ignored.
* **callbacks** (list of [TrainerCallback](https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.TrainerCallback), *optional*, defaults to `None`) —
  List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
  in [here](https://huggingface.co/docs/transformers/main_classes/callback).

  If you want to remove one of the default callbacks used, use the [remove\_callback](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.remove_callback)
  method.
* **optimizers** (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`) —
  A tuple containing the optimizer and the scheduler to use. Will default to an instance of `AdamW` on your
  model and a scheduler given by `get_linear_schedule_with_warmup` controlled by `args`.
* **peft\_config** (`~peft.PeftConfig`, *optional*, defaults to `None`) —
  PEFT configuration used to wrap the model. If `None`, the model is not wrapped.

Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
Models](https://huggingface.co/papers/2402.03300).

Example:

Copied

```
from datasets import load_dataset
from trl import GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")


def reward_func(completions, **kwargs):
    # Dummy reward function that rewards completions with more unique letters.
    return [float(len(set(completion))) for completion in completions]


trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_func,
    train_dataset=dataset,
)

trainer.train()
```

#### train

  [< source >](https://github.com/huggingface/trl/blob/main/transformers/trainer.py#L2131)

( resume\_from\_checkpoint: typing.Union[str, bool, NoneType] = None trial: typing.Union[ForwardRef('optuna.Trial'), dict[str, typing.Any], NoneType] = None ignore\_keys\_for\_eval: typing.Optional[list[str]] = None \*\*kwargs  )

Parameters

* **resume\_from\_checkpoint** (`str` or `bool`, *optional*) —
  If a `str`, local path to a saved checkpoint as saved by a previous instance of `Trainer`. If a
  `bool` and equals `True`, load the last checkpoint in *args.output\_dir* as saved by a previous instance
  of `Trainer`. If present, training will resume from the model/optimizer/scheduler states loaded here.
* **trial** (`optuna.Trial` or `dict[str, Any]`, *optional*) —
  The trial run or the hyperparameter dictionary for hyperparameter search.
* **ignore\_keys\_for\_eval** (`list[str]`, *optional*) —
  A list of keys in the output of your model (if it is a dictionary) that should be ignored when
  gathering predictions for evaluation during the training.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional keyword arguments used to hide deprecated arguments

Main training entry point.

#### save\_model

  [< source >](https://github.com/huggingface/trl/blob/main/transformers/trainer.py#L3939)

( output\_dir: typing.Optional[str] = None \_internal\_call: bool = False  )

Will save the model, so you can reload it using `from_pretrained()`.

Will only save from the main process.

#### push\_to\_hub

  [< source >](https://github.com/huggingface/trl/blob/main/transformers/trainer.py#L4877)

( commit\_message: typing.Optional[str] = 'End of training' blocking: bool = True token: typing.Optional[str] = None revision: typing.Optional[str] = None \*\*kwargs  )

Parameters

* **commit\_message** (`str`, *optional*, defaults to `"End of training"`) —
  Message to commit while pushing.
* **blocking** (`bool`, *optional*, defaults to `True`) —
  Whether the function should return only when the `git push` has finished.
* **token** (`str`, *optional*, defaults to `None`) —
  Token with write permission to overwrite Trainer’s original args.
* **revision** (`str`, *optional*) —
  The git revision to commit from. Defaults to the head of the “main” branch.
* **kwargs** (`dict[str, Any]`, *optional*) —
  Additional keyword arguments passed along to `~Trainer.create_model_card`.

Upload `self.model` and `self.processing_class` to the 🤗 model hub on the repo `self.args.hub_model_id`.

GRPOConfig
----------

### class trl.GRPOConfig

  [< source >](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py#L23)

( output\_dir: typing.Optional[str] = None overwrite\_output\_dir: bool = False do\_train: bool = False do\_eval: bool = False do\_predict: bool = False eval\_strategy: typing.Union[transformers.trainer\_utils.IntervalStrategy, str] = 'no' prediction\_loss\_only: bool = False per\_device\_train\_batch\_size: int = 8 per\_device\_eval\_batch\_size: int = 8 per\_gpu\_train\_batch\_size: typing.Optional[int] = None per\_gpu\_eval\_batch\_size: typing.Optional[int] = None gradient\_accumulation\_steps: int = 1 eval\_accumulation\_steps: typing.Optional[int] = None eval\_delay: typing.Optional[float] = 0 torch\_empty\_cache\_steps: typing.Optional[int] = None learning\_rate: float = 1e-06 weight\_decay: float = 0.0 adam\_beta1: float = 0.9 adam\_beta2: float = 0.999 adam\_epsilon: float = 1e-08 max\_grad\_norm: float = 1.0 num\_train\_epochs: float = 3.0 max\_steps: int = -1 lr\_scheduler\_type: typing.Union[transformers.trainer\_utils.SchedulerType, str] = 'linear' lr\_scheduler\_kwargs: typing.Union[dict[str, typing.Any], str, NoneType] = <factory> warmup\_ratio: float = 0.0 warmup\_steps: int = 0 log\_level: str = 'passive' log\_level\_replica: str = 'warning' log\_on\_each\_node: bool = True logging\_dir: typing.Optional[str] = None logging\_strategy: typing.Union[transformers.trainer\_utils.IntervalStrategy, str] = 'steps' logging\_first\_step: bool = False logging\_steps: float = 10 logging\_nan\_inf\_filter: bool = True save\_strategy: typing.Union[transformers.trainer\_utils.SaveStrategy, str] = 'steps' save\_steps: float = 500 save\_total\_limit: typing.Optional[int] = None save\_safetensors: typing.Optional[bool] = True save\_on\_each\_node: bool = False save\_only\_model: bool = False restore\_callback\_states\_from\_checkpoint: bool = False no\_cuda: bool = False use\_cpu: bool = False use\_mps\_device: bool = False seed: int = 42 data\_seed: typing.Optional[int] = None jit\_mode\_eval: bool = False use\_ipex: bool = False bf16: typing.Optional[bool] = None fp16: bool = False fp16\_opt\_level: str = 'O1' half\_precision\_backend: str = 'auto' bf16\_full\_eval: bool = False fp16\_full\_eval: bool = False tf32: typing.Optional[bool] = None local\_rank: int = -1 ddp\_backend: typing.Optional[str] = None tpu\_num\_cores: typing.Optional[int] = None tpu\_metrics\_debug: bool = False debug: typing.Union[str, list[transformers.debug\_utils.DebugOption]] = '' dataloader\_drop\_last: bool = False eval\_steps: typing.Optional[float] = None dataloader\_num\_workers: int = 0 dataloader\_prefetch\_factor: typing.Optional[int] = None past\_index: int = -1 run\_name: typing.Optional[str] = None disable\_tqdm: typing.Optional[bool] = None remove\_unused\_columns: typing.Optional[bool] = False label\_names: typing.Optional[list[str]] = None load\_best\_model\_at\_end: typing.Optional[bool] = False metric\_for\_best\_model: typing.Optional[str] = None greater\_is\_better: typing.Optional[bool] = None ignore\_data\_skip: bool = False fsdp: typing.Union[list[transformers.trainer\_utils.FSDPOption], str, NoneType] = '' fsdp\_min\_num\_params: int = 0 fsdp\_config: typing.Union[dict[str, typing.Any], str, NoneType] = None fsdp\_transformer\_layer\_cls\_to\_wrap: typing.Optional[str] = None accelerator\_config: typing.Union[dict, str, NoneType] = None deepspeed: typing.Union[dict, str, NoneType] = None label\_smoothing\_factor: float = 0.0 optim: typing.Union[transformers.training\_args.OptimizerNames, str] = 'adamw\_torch' optim\_args: typing.Optional[str] = None adafactor: bool = False group\_by\_length: bool = False length\_column\_name: typing.Optional[str] = 'length' report\_to: typing.Union[NoneType, str, list[str]] = None ddp\_find\_unused\_parameters: typing.Optional[bool] = None ddp\_bucket\_cap\_mb: typing.Optional[int] = None ddp\_broadcast\_buffers: typing.Optional[bool] = None dataloader\_pin\_memory: bool = True dataloader\_persistent\_workers: bool = False skip\_memory\_metrics: bool = True use\_legacy\_prediction\_loop: bool = False push\_to\_hub: bool = False resume\_from\_checkpoint: typing.Optional[str] = None hub\_model\_id: typing.Optional[str] = None hub\_strategy: typing.Union[transformers.trainer\_utils.HubStrategy, str] = 'every\_save' hub\_token: typing.Optional[str] = None hub\_private\_repo: typing.Optional[bool] = None hub\_always\_push: bool = False hub\_revision: typing.Optional[str] = None gradient\_checkpointing: bool = False gradient\_checkpointing\_kwargs: typing.Union[dict[str, typing.Any], str, NoneType] = None include\_inputs\_for\_metrics: bool = False include\_for\_metrics: list = <factory> eval\_do\_concat\_batches: bool = True fp16\_backend: str = 'auto' push\_to\_hub\_model\_id: typing.Optional[str] = None push\_to\_hub\_organization: typing.Optional[str] = None push\_to\_hub\_token: typing.Optional[str] = None mp\_parameters: str = '' auto\_find\_batch\_size: bool = False full\_determinism: bool = False torchdynamo: typing.Optional[str] = None ray\_scope: typing.Optional[str] = 'last' ddp\_timeout: int = 1800 torch\_compile: bool = False torch\_compile\_backend: typing.Optional[str] = None torch\_compile\_mode: typing.Optional[str] = None include\_tokens\_per\_second: typing.Optional[bool] = False include\_num\_input\_tokens\_seen: typing.Optional[bool] = False neftune\_noise\_alpha: typing.Optional[float] = None optim\_target\_modules: typing.Union[NoneType, str, list[str]] = None batch\_eval\_metrics: bool = False eval\_on\_start: bool = False use\_liger\_kernel: typing.Optional[bool] = False liger\_kernel\_config: typing.Optional[dict[str, bool]] = None eval\_use\_gather\_object: typing.Optional[bool] = False average\_tokens\_across\_devices: typing.Optional[bool] = True model\_init\_kwargs: typing.Union[dict, str, NoneType] = None disable\_dropout: bool = False max\_prompt\_length: typing.Optional[int] = 512 num\_generations: typing.Optional[int] = 8 max\_completion\_length: typing.Optional[int] = 256 ds3\_gather\_for\_generation: bool = True shuffle\_dataset: typing.Optional[bool] = True generation\_batch\_size: typing.Optional[int] = None steps\_per\_generation: typing.Optional[int] = None temperature: float = 1.0 top\_p: float = 1.0 top\_k: typing.Optional[int] = None min\_p: typing.Optional[float] = None generation\_kwargs: typing.Optional[dict] = None repetition\_penalty: float = 1.0 use\_transformers\_paged: bool = False cache\_implementation: typing.Optional[str] = None use\_vllm: bool = False vllm\_server\_base\_url: typing.Optional[str] = None vllm\_mode: str = 'server' vllm\_model\_impl: str = 'vllm' vllm\_guided\_decoding\_regex: typing.Optional[str] = None vllm\_server\_host: str = '0.0.0.0' vllm\_server\_port: int = 8000 vllm\_server\_timeout: float = 240.0 vllm\_gpu\_memory\_utilization: float = 0.3 vllm\_tensor\_parallel\_size: int = 1 beta: float = 0.0 num\_iterations: int = 1 epsilon: float = 0.2 delta: typing.Optional[float] = None epsilon\_high: typing.Optional[float] = None importance\_sampling\_level: str = 'token' reward\_weights: typing.Optional[list[float]] = None scale\_rewards: bool = True loss\_type: str = 'bnpo' mask\_truncated\_completions: bool = False sync\_ref\_model: bool = False ref\_model\_mixup\_alpha: float = 0.6 ref\_model\_sync\_steps: int = 512 top\_entropy\_quantile: float = 1.0 use\_liger\_loss: bool = False log\_completions: bool = False num\_completions\_to\_print: typing.Optional[int] = None wandb\_log\_unique\_prompts: typing.Optional[bool] = False  )

Parameters that control the model and reference model

* **model\_init\_kwargs** (`str`, `dict[str, Any]` or `None`, *optional*, defaults to `None`) —
  Keyword arguments for [from\_pretrained](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained), used when the `model`
  argument of the [GRPOTrainer](/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer) is provided as a string.
* **disable\_dropout** (`bool`, *optional*, defaults to `False`) —
  Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents
  the model from generating different logprobs for the same input.

Parameters that control the data preprocessing

* **remove\_unused\_columns** (`bool`, *optional*, defaults to `False`) —
  Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
  requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
* **max\_prompt\_length** (`int` or `None`, *optional*, defaults to `512`) —
  Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
* **num\_generations** (`int` or `None`, *optional*, defaults to `8`) —
  Number of generations per prompt to sample. The effective batch size (num\_processes \* per\_device\_batch\_size
  + gradient\_accumulation\_steps) must be evenly divisible by this value.
* **max\_completion\_length** (`int` or `None`, *optional*, defaults to `256`) —
  Maximum length of the generated completion.
* **ds3\_gather\_for\_generation** (`bool`, *optional*, defaults to `True`) —
  This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
  improving generation speed. However, disabling this option allows training models that exceed the VRAM
  capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
  with vLLM generation.
* **shuffle\_dataset** (`bool`, *optional*, defaults to `True`) —
  Whether to shuffle the training dataset.

Parameters that control generation

* **generation\_batch\_size** — (`int` or `None`, *optional*, defaults to `None`):
  Batch size to use for generation. If `None`, it defaults to the effective training batch size:
  `per_device_train_batch_size * num_processes * steps_per_generation`. In other words, there is one
  generation batch processed per optimization step. Mutually exclusive with `steps_per_generation`.
* **steps\_per\_generation** — (`int` or `None`, *optional*, defaults to `None`):
  Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. Mutually exclusive
  with `generation_batch_size`.
* **temperature** (`float`, defaults to `1.0`) —
  Temperature for sampling. The higher the temperature, the more random the completions.
* **top\_p** (`float`, *optional*, defaults to `1.0`) —
  Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
  `1.0` to consider all tokens.
* **top\_k** (`int` or `None`, *optional*, defaults to `None`) —
  Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
  disabled and all tokens are considered.
* **min\_p** (`float` or `None`, *optional*, defaults to `None`) —
  Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
  value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
* **repetition\_penalty** (`float`, *optional*, defaults to `1.0`) —
  Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
  Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
  tokens.
* **use\_transformers\_paged** (`bool`, *optional*, defaults to `False`) —
  Whether to use the `transformers` paged implementation for generation. If set to `True`, the `transformers`
  paged implementation will be used for generation instead of the default padded implementation. This
  parameter is only effective when `use_vllm` is set to `False`.
* **cache\_implementation** (`str` or `None`, *optional*, defaults to `None`) —
  Implementation of the cache method for faster generation when `use_vllm` is set to `False`.
* **generation\_kwargs** (`dict[str, Any]` or `None`, *optional*, defaults to `None`) —
  Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or `SamplingParams` (if
  using vLLM) when sampling completions. This can be used to further customize the generation behavior, such
  as setting `supress_tokens`, `num_beams`, etc. If it contains keys that conflict with the other generation
  parameters (like `min_p`, `top_p`, etc.), they will override them.

Parameters that control generation acceleration powered by vLLM

* **use\_vllm** (`bool`, *optional*, defaults to `False`) —
  Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
  instead of the default model.generate(). Requires `vllm` to be installed.
* **vllm\_mode** (`str`, *optional*, defaults to `"server"`) —
  Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `"server"` or
  `"colocate"`.
  + `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
    server is running (start with `trl vllm-serve`).
  + `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
    separate server but may cause resource contention with training.
* **vllm\_guided\_decoding\_regex** (`str` or `None`, *optional*, defaults to `None`) —
  Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

Parameters that control the vLLM server (only used when `vllm\_mode` is `"server"`)

* **vllm\_server\_base\_url** (`str` or `None`, *optional*, defaults to `None`) —
  Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `vllm_server_host` and
  `vllm_server_port` are ignored.
* **vllm\_server\_host** (`str`, *optional*, defaults to `"0.0.0.0"`) —
  Host of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
* **vllm\_server\_port** (`int`, *optional*, defaults to `8000`) —
  Port of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
* **vllm\_server\_timeout** (`float`, *optional*, defaults to `240.0`) —
  Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
  timeout, a `ConnectionError` is raised.

Parameters that control colocated vLLM execution (only used when `vllm\_mode` is `"colocate"`)

* **vllm\_gpu\_memory\_utilization** (`float`, *optional*, defaults to `0.3`) —
  Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
  `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
  launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
* **vllm\_tensor\_parallel\_size** (`int`, *optional*, defaults to `1`) —
  Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
  `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
  launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
* **vllm\_model\_impl** (`str`, *optional*, defaults to `"vllm"`) —
  Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
  the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
  implementation.

Parameters that control the training

* **beta** (`float`, *optional*, defaults to `0.0`) —
  KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving
  training speed.
* **num\_iterations** (`int`, *optional*, defaults to `1`) —
  Number of iterations per batch (denoted as μ in the algorithm).
* **epsilon** (`float`, *optional*, defaults to `0.2`) —
  Epsilon value for clipping.
* **delta** — (`float` or `None`, *optional*, defaults to `None`):
  Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` (default), standard
  GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This method is introduced in
  the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291).
* **epsilon\_high** (`float` or `None`, *optional*, defaults to `None`) —
  Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
  specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.
* **importance\_sampling\_level** (`str`, *optional*, defaults to `"token"`) —
  Controls whether importance sampling ratios are computed at the `"token"` or `"sequence"` level. `"token"`
  keeps the raw per-token log-probability ratios (one weight per token). `"sequence"` averages the
  log-probability ratios across valid tokens to produce a single ratio per sequence. The
  [GSPO paper](https://huggingface.co/papers/2507.18071) shows that sequence-level sampling often yields more
  stable training and better alignment with sequence-level rewards.
* **reward\_weights** (`list[float]` or `None`, *optional*, defaults to `None`) —
  Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
  weighted equally with weight `1.0`.
* **scale\_rewards** (`bool`, *optional*, defaults to `True`) —
  Whether to scale the rewards by dividing them by their standard deviation. If `True` (default), the rewards
  are normalized by the standard deviation, ensuring they have unit variance. If `False`, no scaling is
  applied. The [Dr. GRPO paper](https://huggingface.co/papers/2503.20783) recommends not scaling the rewards,
  as scaling by the standard deviation introduces a question-level difficulty bias.
* **loss\_type** (`str`, *optional*, defaults to `"bnpo"`) —
  Specifies the loss formulation to use. Supported values are:
  + `"grpo"`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to
    length bias—this approach tends to prefer shorter completions with positive advantages and longer ones
    with negative advantages.
  + `"bnpo"`: Aggregates token-level losses by normalizing number of active token in the local batch.
    Note that normalization is performed over the local batch only, so results may slightly vary depending
    on the local batch size, despite a constant effective batch size. When using
    `per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss.
  + `"dr_grpo"`: Aggregates token-level losses by normalizing with a global constant. This method was
    introduced in the [Dr. GRPO paper](https://huggingface.co/papers/2503.20783) to eliminate length bias.
    The value of the constant corresponds to `max_completion_length`.
* **mask\_truncated\_completions** (`bool`, *optional*, defaults to `False`) —
  When enabled, truncated completions are excluded from the loss calculation, preventing them from being
  incorrectly penalized and introducing noise during training. According to the
  [DAPO](https://huggingface.co/papers/2503.14476) paper, this is a good practice for training stability.
* **sync\_ref\_model** (`bool`, *optional*, defaults to `False`) —
  Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
  the `ref_model_mixup_alpha` parameter. This synchronization originates from the
  [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
* **ref\_model\_mixup\_alpha** (`float`, *optional*, defaults to `0.6`) —
  α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
  between the current policy and the previous reference policy during updates. The reference policy is
  updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
  must set `sync_ref_model=True`.
* **ref\_model\_sync\_steps** (`int`, *optional*, defaults to `512`) —
  τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
  frequently the current policy is synchronized with the reference policy. To use this parameter, you must
  set `sync_ref_model=True`.
* **top\_entropy\_quantile** (`float`, *optional*, defaults to `1.0`) —
  ρ parameter from [Beyond the 80/20 Rule](https://huggingface.co/papers/2506.01939). Keeps in the policy
  loss term only the top-ρ quantile of tokens by entropy of the probability distribution at each sequence
  position, improving results. Range: `[0.0-1.0]`. A value of `0.0` masks all but the highest entropy token;
  `1.0` keeps all tokens. The paper recommends a value of `0.2`.
  If used with `mask_truncated_completions=True`, only tokens from non-truncated completions are considered.
* **use\_liger\_loss** (`bool`, *optional*, defaults to `False`) —
  Whether to use the Liger GRPO loss.

Parameters that control the logging

* **log\_completions** (`bool`, *optional*, defaults to `False`) —
  Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed,
  it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
* **num\_completions\_to\_print** (`int` or `None`, *optional*, defaults to `None`) —
  Number of completions to print with `rich`. If `None`, all completions are logged.
* **wandb\_log\_unique\_prompts** (`bool`, *optional*, defaults to `False`) —
  Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, all prompts
  are logged.

Configuration class for the [GRPOTrainer](/docs/trl/main/en/grpo_trainer#trl.GRPOTrainer).

This class includes only the parameters that are specific to GRPO training. For a full list of training arguments,
please refer to the [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) documentation. Note that default values in this class may
differ from those in [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments).

Using [HfArgumentParser](https://huggingface.co/docs/transformers/main/en/internal/trainer_utils#transformers.HfArgumentParser) we can turn this class into
[argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
command line.

[< > Update on GitHub](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md)

[←GKD](/docs/trl/main/en/gkd_trainer)
[KTO→](/docs/trl/main/en/kto_trainer)

[GRPO Trainer](#grpo-trainer)
[Overview](#overview)
[Quick start](#quick-start)
[Looking deeper into the GRPO method](#looking-deeper-into-the-grpo-method)
[Generating completions](#generating-completions)
[Computing the advantage](#computing-the-advantage)
[Estimating the KL divergence](#estimating-the-kl-divergence)
[Computing the loss](#computing-the-loss)
[Loss Types](#loss-types)
[Logged metrics](#logged-metrics)
[Customization](#customization)
[Speed up training with vLLM-powered generation](#speed-up-training-with-vllm-powered-generation)
[🔌 Option 1: Server mode](#-option-1-server-mode)
[🧩 Option 2: Colocate mode](#-option-2-colocate-mode)
[GRPO at scale: train a 70B+ Model on multiple nodes](#grpo-at-scale-train-a-70b-model-on-multiple-nodes)
[Using a custom reward function](#using-a-custom-reward-function)
[Example 1: Reward longer completions](#example-1-reward-longer-completions)
[Example 1.1: Reward longer completions (based in the number of characters)](#example-11-reward-longer-completions-based-in-the-number-of-characters)
[Example 2: Reward completions with specific format](#example-2-reward-completions-with-specific-format)
[Example 3: Reward completions based on a reference](#example-3-reward-completions-based-on-a-reference)
[Example 4: Multi-task reward functions](#example-4-multi-task-reward-functions)
[Passing the reward function to the trainer](#passing-the-reward-function-to-the-trainer)
[Vision-Language Model (VLM) Training](#vision-language-model-vlm-training)
[Supported Models](#supported-models)
[Quick Start](#quick-start)
[Configuration Tips](#configuration-tips)
[Dataset Format](#dataset-format)
[GRPOTrainer](#trl.GRPOTrainer)
[GRPOConfig](#trl.GRPOConfig)