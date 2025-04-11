# MoEMeta

This repository implements **MoEMeta**, a Mixture-of-Experts based meta-learning model for few-shot relational learning on knowledge graphs.

## 📁 File Overview

| File               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `main.py`          | Entry point. Loads data, builds model, runs training/testing. |
| `trainer.py`       | Defines the MAML-style meta-training and testing process.    |
| `params.py`        | Parses command-line arguments and hyperparameters.           |
| `data_loader.py`   | Loads datasets and prepares support/query sets.              |
| `models.py`        | Defines the overall MoEMeta architecture.                    |
| `moe.py`           | Implements the Mixture-of-Experts module and sparse gating.  |
| `embedding.py`     | Handles entity and relation embeddings.                      |
| `requirements.txt` | Python dependencies.                                         |
| `.gitignore`       | Specifies ignored files.                                     |

## 🔧 Setup

```bash
pip install -r requirements.txt
```

## 🔧 Train and test examples:

Train MoEMeta on Nell-One under 1-shot setting:

```
python main.py --dataset NELL-One --few 1 --prefix example-train-nell --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 0 --step train
```

Test MoEMeta on Nell-One under 1-shot setting:

```
python main.py --dataset NELL-One --few 1 --prefix example-test-nell --learning_rate 0.001 --checkpoint_epoch 1000 --eval_epoch 1000 --batch_size 1024 --device 0 --step test
```

