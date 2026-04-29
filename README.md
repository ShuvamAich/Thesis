# Deception Detection Experiment Report

This README documents the full experiment trail carried out in this workspace across two related projects:

- `c:\Thesis\deception_detection`: cue-aware pipeline using sampled video frames plus prompt-engineered behavioural evidence derived from the dataset CSV
- `c:\Thesis\deception_detection_video_only`: video-only baseline using sampled frames without transcript text, CSV gesture annotations, or handcrafted deception cues

The main goal across the session was to classify each clip in the Real-life Deception Detection 2016 dataset as `truthful` or `deceptive` using local open-source multimodal models that fit an RTX A3000 12 GB laptop GPU.

## Dataset

- Dataset: Real-life Deception Detection 2016
- Total clips: 121
- Deceptive clips: 61
- Truthful clips: 60
- Available sources in the dataset:
	- video clips
	- transcripts
	- gesture annotation CSV

Unless noted otherwise, the full-run experiments below used all 121 clips and 1 sampled frame per video.

## Runtime Environment

- OS: Windows
- Python: `C:/Python312/python.exe`
- GPU: NVIDIA RTX A3000 Laptop GPU, 12 GB VRAM
- Torch: CUDA-enabled `torch 2.11.0+cu128`
- Transformers: source build `5.7.0.dev0`

## Model Notes

The following models were used or evaluated during the session:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`

The larger `Qwen3-VL-235B-A22B` family was checked but not used locally because it is not practical on the available 12 GB GPU.

## Project Summary

### 1. Cue-Aware Project

Path: `c:\Thesis\deception_detection`

This project uses:

- sampled video frames
- prompt-engineered behavioural evidence produced from the CSV gesture annotations
- iterative prompt remodeling to reduce class collapse and improve balance between truthful and deceptive predictions

### 2. Video-Only Project

Path: `c:\Thesis\deception_detection_video_only`

This project uses:

- sampled video frames only
- no transcript text
- no CSV gesture annotations
- no handcrafted behavioural evidence

## Experiment Log

### Experiment 1: Initial cue-aware baseline

- Project: `deception_detection`
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Inputs: video frames with the original prompt design
- Results directory: `c:\Thesis\deception_detection\results_full_121`
- Metrics:
	- Accuracy: `0.5041`
	- Macro-F1: `0.3352`
	- AUC-ROC: `0.5000`
- Inference:
	- the model effectively collapsed toward `deceptive`
	- truthful recall was effectively zero
	- this established the need for prompt redesign

### Experiment 2: Hand, face, gaze only prompt

- Project: `deception_detection`
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Inputs: video frames plus only hand, face, and gaze cues, with transcript removed
- Results directory: `c:\Thesis\deception_detection\results_hfg_only_121`
- Metrics:
	- Accuracy: `0.5041`
	- Macro-F1: `0.3352`
	- AUC-ROC: `0.5000`
- Inference:
	- removing transcript and restricting cues did not improve class separation
	- the model still collapsed toward `deceptive`

### Experiment 3: Balanced cue-aware prompt, 3B model

- Project: `deception_detection`
- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Inputs: video frames plus more balanced cue wording
- Results directory: `c:\Thesis\deception_detection\results_hfg_balanced_121_3b_nf1`
- Metrics:
	- Accuracy: `0.5041`
	- Macro-F1: `0.3495`
	- AUC-ROC: `0.5163`
- Inference:
	- prompt balancing reduced the worst bias slightly
	- the run still heavily favored `deceptive`
	- 3B capacity remained a limiting factor

### Experiment 4: Balanced cue-aware prompt, 7B model

- Project: `deception_detection`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Inputs: same balanced cue-aware prompt, larger model, 4-bit loading
- Results directory: `c:\Thesis\deception_detection\results_hfg_balanced_121_7b_nf1`
- Metrics:
	- Accuracy: `0.5289`
	- Macro-F1: `0.4319`
	- AUC-ROC: `0.5481`
- Inference:
	- moving to 7B produced a clear improvement over 3B
	- however, the bias flipped strongly toward `truthful`
	- truthful recall became very high while deceptive recall dropped sharply

### Experiment 5: All-gesture tuned prompt, 7B model

- Project: `deception_detection`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Inputs: all gesture families plus stronger tension and avoidance weighting
- Results directory: `c:\Thesis\deception_detection\results_hfg_allgestures_tuned_121_7b_nf1`
- Metrics:
	- Accuracy: `0.4132`
	- Macro-F1: `0.3562`
	- AUC-ROC: `0.4990`
- Inference:
	- the stronger prompt over-corrected toward `deceptive`
	- deceptive recall improved, but truthful recall fell heavily
	- the cue weighting was too aggressive

### Experiment 6: All-gesture moderated prompt, 7B model

- Project: `deception_detection`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Inputs: all gesture families plus moderated cue-balance wording
- Results directory: `c:\Thesis\deception_detection\results_hfg_allgestures_moderated_121_7b_nf1`
- Metrics:
	- Accuracy: `0.4876`
	- Macro-F1: `0.4848`
	- AUC-ROC: `0.6434`
- Inference:
	- this became the best 7B prompt variant overall
	- class balance improved substantially relative to earlier runs
	- it achieved the strongest AUC of the entire session
	- it remained slightly more balanced than earlier 7B prompt variants, though still imperfect

### Experiment 7: Moderated prompt 20-sample subset check

- Project: `deception_detection`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Inputs: moderated all-gesture prompt
- Samples: 20 balanced subset
- Metrics observed during the session:
	- Accuracy: `0.7500`
	- Macro-F1: `0.7442`
	- AUC-ROC: `0.6800`
- Inference:
	- the moderated prompt looked promising on a balanced subset
	- the full 121-sample run did not preserve that same level of performance
	- this highlighted the gap between small validation subsets and full-dataset behavior

### Experiment 8: Qwen3-VL-8B cue-aware run

- Project: `deception_detection`
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Inputs: video frames plus the moderated cue-aware prompt
- Results directory: `c:\Thesis\deception_detection\results_qwen3vl8b_121`
- Metrics:
	- Accuracy: `0.5207`
	- Macro-F1: `0.5132`
	- AUC-ROC: `0.6277`
- Inference:
	- this became the best overall run by Macro-F1 and by practical class balance
	- it improved over the best 7B Qwen2.5 run on accuracy and Macro-F1
	- its AUC was slightly below the moderated 7B cue-aware run, but the overall classification behavior was stronger and more usable

### Experiment 9: Video-only smoke test

- Project: `deception_detection_video_only`
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Inputs: sampled video frames only
- Results directory: `c:\Thesis\deception_detection_video_only\results_subset1`
- Sample result:
	- `trial_lie_001.mp4` predicted `truthful` with confidence `0.75`
- Inference:
	- the video-only project executed correctly end to end
	- even the first smoke test suggested that removing cue engineering could hurt deception sensitivity

### Experiment 10: Video-only strict one-word prompt

- Project: `deception_detection_video_only`
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Inputs: sampled video frames only, with a strict one-word output prompt adapted from a transcript-style template
- Results directory: `c:\Thesis\deception_detection_video_only\results_qwen3vl8b_121_strictprompt`
- Metrics:
	- Accuracy: `0.5124`
	- Macro-F1: `0.3669`
	- AUC-ROC: `0.5000`
- Inference:
	- the strict prompt caused a strong collapse toward `truthful`
	- truthful recall reached `1.00`, but deceptive recall collapsed to `0.03`
	- the prompt style was too restrictive and removed useful uncertainty structure from the model output

### Experiment 11: Cue-aware 3-label run

- Project: `deception_detection_3label`
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Inputs: sampled video frames plus cue-aware prompt engineering, with predictions expanded to `truthful`, `deceptive`, or `dont know`
- Results directory: `c:\Thesis\deception_detection_3label\results_qwen3vl8b_121`
- Metrics:
	- Accuracy: `0.2066`
	- Macro-F1: `0.1814`
	- Balanced Accuracy: `0.2053`
- Prediction counts:
	- `truthful`: `14`
	- `deceptive`: `34`
	- `dont know`: `73`
- Inference:
	- allowing a third abstention-style label caused the cue-aware system to overuse `dont know`
	- this sharply reduced practical classification performance on the binary-ground-truth dataset
	- the cue-aware prompt was much more uncertainty-sensitive than the binary version, but that uncertainty was not well calibrated

### Experiment 12: Video-only 3-label run

- Project: `deception_detection_video_only_3label`
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Inputs: sampled video frames only, with predictions expanded to `truthful`, `deceptive`, or `dont know`
- Results directory: `c:\Thesis\deception_detection_video_only_3label\results_qwen3vl8b_121`
- Metrics:
	- Accuracy: `0.5041`
	- Macro-F1: `0.2355`
	- Balanced Accuracy: `0.5082`
- Prediction counts:
	- `truthful`: `118`
	- `deceptive`: `1`
	- `dont know`: `2`
- Inference:
	- the video-only 3-label variant barely used the third label at all
	- it effectively collapsed to `truthful`, which preserved headline accuracy near chance but produced weak class discrimination
	- compared with the cue-aware 3-label run, it was less uncertain but also much less sensitive to deceptive clips

## Consolidated Results Table

| Experiment | Project | Model | Input style | Accuracy | Macro-F1 | AUC-ROC | Balanced Acc. | Main behavior |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Initial baseline | cue-aware | Qwen2.5-VL-3B | original cue-aware | 0.5041 | 0.3352 | 0.5000 | n/a | collapsed to deceptive |
| HFG only | cue-aware | Qwen2.5-VL-3B | hand/face/gaze only | 0.5041 | 0.3352 | 0.5000 | n/a | collapsed to deceptive |
| Balanced 3B | cue-aware | Qwen2.5-VL-3B | balanced cue prompt | 0.5041 | 0.3495 | 0.5163 | n/a | still deceptive bias |
| Balanced 7B | cue-aware | Qwen2.5-VL-7B | balanced cue prompt | 0.5289 | 0.4319 | 0.5481 | n/a | shifted to truthful bias |
| All-gesture tuned 7B | cue-aware | Qwen2.5-VL-7B | strong all-gesture weighting | 0.4132 | 0.3562 | 0.4990 | n/a | over-corrected to deceptive |
| All-gesture moderated 7B | cue-aware | Qwen2.5-VL-7B | moderated all-gesture prompt | 0.4876 | 0.4848 | 0.6434 | n/a | best 7B balance and best AUC |
| Qwen3-VL-8B cue-aware | cue-aware | Qwen3-VL-8B | moderated cue-aware prompt | 0.5207 | 0.5132 | 0.6277 | n/a | best overall Macro-F1 |
| Video-only strict prompt | video-only | Qwen3-VL-8B | frames only, one-word output | 0.5124 | 0.3669 | 0.5000 | n/a | collapsed to truthful |
| Qwen3-VL-8B cue-aware 3-label | cue-aware | Qwen3-VL-8B | cue-aware prompt with truthful/deceptive/dont know | 0.2066 | 0.1814 | n/a | 0.2053 | heavy dont know usage, weak overall discrimination |
| Qwen3-VL-8B video-only 3-label | video-only | Qwen3-VL-8B | frames only with truthful/deceptive/dont know | 0.5041 | 0.2355 | n/a | 0.5082 | near-complete truthful bias |

## Main Inferences

### 1. Prompt wording matters as much as model size

The early runs showed that poor prompt balance produced immediate class collapse. The system could be pushed toward either `deceptive` or `truthful` just by how the behavioural evidence was framed.

### 2. Moving from 3B to 7B helped, but did not solve the problem by itself

The 7B model improved performance, but it still remained highly sensitive to prompt bias. Capacity helped, but the prompt architecture still controlled class behavior strongly.

### 3. Moderated cue integration worked better than either weak cues or aggressive cues

The strongest cue-aware results came from using all gesture families while explicitly preventing a single cue or summary count from dominating the decision.

### 4. Qwen3-VL-8B was the strongest practical local model in this session

It fit the hardware with 4-bit loading, gave the best full-run Macro-F1, and maintained good overall balance. It became the best local baseline achieved in the workspace.

### 5. Video-only prompting was clearly weaker than cue-aware prompting

The strict video-only run had similar headline accuracy to some weaker cue-based runs, but this was misleading. Macro-F1 and AUC showed that the model was not truly distinguishing the classes well. It defaulted heavily to `truthful`, which made the output far less reliable.

### 6. One-word output constraints reduced useful calibration

Forcing the model to output only `Truthful` or `Deceptive` simplified parsing, but it also removed confidence structure and encouraged trivial majority-like behavior.

## Best Configurations Reached in This Session

### Best overall practical run

- Project: `deception_detection`
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Results: `c:\Thesis\deception_detection\results_qwen3vl8b_121`
- Why it mattered:
	- best Macro-F1 of the session on the full 121-sample dataset
	- better overall balance than the earlier 3B and 7B runs
	- practical on the available local GPU

### Best 7B prompt-engineering run

- Project: `deception_detection`
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Results: `c:\Thesis\deception_detection\results_hfg_allgestures_moderated_121_7b_nf1`
- Why it mattered:
	- strongest AUC in the session
	- showed that moderated full-cue integration was a better strategy than either sparse cues or strongly weighted cues

## Reproducing the Current Video-Only Project

From `c:\Thesis\deception_detection_video_only`:

```powershell
$env:HF_TOKEN='YOUR_TOKEN'
$env:HUGGINGFACE_HUB_TOKEN='YOUR_TOKEN'
$env:TRANSFORMERS_VERBOSITY='error'
$env:HF_HUB_VERBOSITY='error'
C:/Python312/python.exe -u .\main.py --num-frames 1 --output-dir 'c:\Thesis\deception_detection_video_only\results_qwen3vl8b_121_strictprompt'
```

## Recommended Next Steps

1. Keep `Qwen/Qwen3-VL-8B-Instruct` as the local baseline model.
2. If the goal is strongest local performance, continue from the cue-aware project rather than the strict video-only branch.
3. If a video-only baseline is still required, redesign the prompt to allow brief structured reasoning instead of a one-word answer.
4. If transcript-only evaluation is required, build a separate transcript pipeline instead of forcing a transcript-style prompt onto frame-only inputs.