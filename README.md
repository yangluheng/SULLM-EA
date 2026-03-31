# SULLM-EA



Most existing EA methods either:
- rely heavily on **embedding similarity computation** based on graph representation learning, or  
- simply incorporate **large language models as post-hoc filtering or reranking modules**, treating KG structures as plain text.

These paradigms fail to fully exploit the structural semantics of KGs and limit the reasoning capability of LLMs in EA.

**SULLM-EA** is an LLM-driven entity alignment framework that explicitly bridges KGs and LLMs by projecting structured entity semantics into the LLM token space, enabling structure-aware alignment reasoning **without relying on embedding similarity computation**.

---

## ✨ Key Features

- **LLM-driven EA without embedding similarity**  
  EA-LLM performs entity alignment directly via LLM reasoning, instead of nearest-neighbor matching in embedding space.

- **Attribute-Relation Mixer**  
  A residual fusion module that combines:
  - relation-aware KG embeddings  
  - attribute description embeddings  

- **KG-LLM Aligner (No Fine-tuning)**  
  Aligns entity embeddings with LLM token space through a lightweight projector, enabling structure-aware reasoning **without fine-tuning the LLM**.

- **Unified Prompting for EA**  
  Attribute summaries are explicitly injected into both projector tuning and LLM reasoning via a unified prompt template.
---

## 📁 Project Structure

```text
.
├── mixer/          # Training Attribute-Relation Mixer
├── aligner/        # Training KG-LLM Aligner
├── bash/           # Bash scripts for training and evaluation
├── data/
│   ├── instruction/   # Instruction templates for EA tasks
│   ├── kg/            # Knowledge graph data
│   └── llm/           # PCA-reduced LLM embeddings
├── results/        # Experimental results and logs
└── saved_model/    # Trained models
```

## 🔧 Environment Setup


### 1️⃣ Create Virtual Environment

```bash
conda create -n ea-llm python=3.9 -y
conda activate ea-llm
```
### 2️⃣📦 Export Dependencies

After setting up the environment and installing all required packages, export the dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 Running SULLM-EA

To train and evaluate SULLM-EA on the ICEWS–WIKI dataset, run:

```bash
bash ./bash/wiki_train_test.sh
```
# Acknowledgements
Our work is based on the existing work of the combination of LLM and GNN, and the code is also written based on **[LLaGA](https://github.com/VITA-Group/LLaGA)** and **[TEA-GLM](https://github.com/W-rudder/TEA-GLM)** these work. Thanks for their open source codes and great work.
