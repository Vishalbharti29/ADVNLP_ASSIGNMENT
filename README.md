# ADVNLP_ASSIGNMENT
# BERT Quantization on Emotion Classification (ANLP Assignment-2)
### Baseline Fine-Tuning • PTQ • QAT • QLoRA

This repository implements all components required in the assignment:

- Baseline fine-tuning of **BERT-base-uncased** on `dair-ai/emotion`
- **Post-Training Quantization (PTQ)** using dynamic int8
- **Quantization-Aware Training (QAT)** with fake quantization
- **QLoRA** 4-bit NF4 fine-tuning (low-rank adapters)
- Evaluation of all four variants with:
  - Macro F1  
  - Accuracy  
  - Per-class F1 & confusion matrix  
  - Model size (MB)  
  - Inference latency (ms/example)  
- Final comparison table: **Baseline vs PTQ vs QAT vs QLoRA**

All models are trained offline and their weights are stored in this repo under `weights/`.  
The **evaluation script/notebook only loads from GitHub and runs inference**, which fits well within **≤ 15 minutes on a Colab GPU**, as required.

---

## 0. Exact Commands to Reproduce Results (Colab, ≤ 15 minutes)

This section satisfies the assignment requirement:

These commands **only run inference & evaluation** using pre-saved weights.  
Training is **not** rerun.

### Step 0 – Open Colab with GPU

1. Go to [Google Colab](https://colab.research.google.com)
2. Click **Runtime → Change runtime type**
3. Set:
   - **Runtime type**: Python 3
   - **Hardware accelerator**: GPU (T4 / L4 / V100)
4. Click **Save**.

---
 Step 1 – Clone this repository in Colab

In a Colab **code cell**, run:

!git clone https://github.com/Vishalbharti29/ADVNLP_ASSIGNMENT.git
%cd ADVNLP_ASSIGNMENT

step 2 -install dependencies-

!pip install -r requirements.txt
!pip install transformers datasets peft bitsandbytes accelerate scikit-learn torch

Step 3 – Run the evaluation script (all 4 variants)
!python codesub.py --mode eval_all

Step 3b – Run evaluation per variant
# Baseline FP32/FP16
!python codesub.py --mode eval --variant baseline

# PTQ (dynamic int8, CPU)
!python codesub.py --mode eval --variant ptq

# QAT (fake quant model)
!python codesub.py --mode eval --variant qat

# QLoRA (4-bit base + adapters)
!python codesub.py --mode eval --variant qlora


