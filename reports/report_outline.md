# Report Outline

## 1. Introduction

- Research goal: analyze how a Transformer language model represents truth and falsehood in fact-verification prompts.
- Chosen phenomenon: capital fact verification, with broader cross-domain comparison.
- Project framing: Locate key layers, Steer/Improve behavior through activation intervention, and reproduce core ideas from recent mechanistic interpretability work on linear truth representations.

## 2. Background

- Transformer residual stream
- Multi-head attention and MLP blocks
- Hook mechanisms in TransformerLens
- Linear probes and logit differences
- Activation patching
- Steering vectors and vector arithmetic

## 3. Reproduction Target

- Reproduce the core claim that truth/falsehood can be linearly readable from model activations in some settings.
- Compare whether the direction is stable across domains and prompts.
- Test whether the discovered direction has causal force through steering or patching.

## 4. Experimental Setup

- Model: GPT-2-small via TransformerLens.
- Dataset: 528 English fact-verification statements, balanced between true and false.
- Domains: capital, continent, element_symbol, book_author, landmark_country, science, math.
- Hook point: `blocks.{layer}.hook_resid_post`.
- Primary metrics: accuracy, AUC, separability AUC, logit difference.

## 5. Locate Results

- Mixed-domain probe results.
- Domain-wise probe sweep.
- Focused capital-fact probe result.
- Interpretation of which layers contain the strongest linear signal.

## 6. Steering Results

- Mean-difference truth direction.
- Probe-direction steering.
- Alpha sweep.
- Current negative result: steering controls internal probe score but does not improve true/false output accuracy under naive global intervention.

## 7. Next Causal Experiments

已完成初版 activation patching：

- Capital recall prompt.
- Patch `hook_resid_post` from clean country prompt into corrupt country prompt.
- Measure recovery of clean capital vs corrupt capital logit difference.
- Module comparison now includes `hook_resid_post`, `hook_attn_out`, and `hook_mlp_out`.

下一步：

- Optional ablation of truth direction.
- Head-level attention patching.

## 8. Discussion

- Why mixed-domain truth representation is weaker than domain-specific representation.
- Whether this supports a universal truth direction or a task-local truth direction.
- Limitations of GPT-2-small and true/false prompting.
- Next steps with Qwen2.5 or instruction-tuned models.

## 9. Conclusion

- Summarize Locate findings.
- Summarize intervention results and limitations.
- State what was reproduced and what remains inconclusive.
