# KaggleOps 🧪🎯

**KaggleOps** is a modular, MLflow-integrated framework designed to streamline and unify your Kaggle competition workflow across multiple modalities — including image, audio, and text tasks.\
Whether you're tackling a computer vision challenge or the latest BirdCLEF audio event, KaggleOps helps you stay organized, reproducible, and competitive. 🏆

---

## 🌟 Key Features

- **Multi-Modality Support**:
  - Image: EfficientNet, ViT, etc.
  - Audio: PANNs, AST, YAMNet
  - Text: BERT, RoBERTa, etc.
- **Task-specific Data Loaders & Preprocessors**
- **Config-based Pipeline Execution**
- **Built-in MLflow Logging & Model Registry**
- **Reusable and Lightweight Codebase**
- **Pluggable Model Components** using PyTorch + timm + HuggingFace

---

## 🧱 Project Structure

```bash
KaggleOps/
├── configs/                  # YAML config files for each task
├── data/
│   ├── loaders/              # Task-specific dataloaders
│   └── preprocessors/        # Augmentation, transforms
├── models/                   # Modular model definitions
├── pipelines/                # Train/infer pipelines
├── mlflow_utils/             # Logging, model registration
├── utils/                    # Metrics, scheduler, helpers
├── main.py                   # Entry point
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/miao4ai/KaggleOps.git
cd KaggleOps
pip install -r requirements.txt
```

### 2. Train a Model

```bash
python main.py --config configs/image_classification.yaml
```

### 3. Track with MLflow

Make sure MLflow server is running:

```bash
mlflow ui
```

Check your metrics, losses, and models at [localhost:5000](http://localhost:5000) 🌛

---

## 🧠 Supported Tasks

| Type  | Example Competition    | Loader            | Backbone Models    |
| ----- | ---------------------- | ----------------- | ------------------ |
| Image | Cassava, AnimalCLEF    | `image_loader.py` | ViT, EfficientNet  |
| Audio | BirdCLEF, ESC-50       | `audio_loader.py` | PANNs, YAMNet, AST |
| Text  | Jigsaw, Toxic Comments | `text_loader.py`  | BERT, RoBERTa      |

---

## 📖 Configuration Style

Each task uses a YAML config:

```yaml
task: image_classification
model:
  name: vit_base_patch32_224
  pretrained: true
training:
  epochs: 10
  batch_size: 32
  optimizer: adam
mlflow:
  enabled: true
  experiment_name: animalclef-vit
```

---

## ❤️ Maintainers

Developed by\
[@miao4ai](https://github.com/miao4ai) — KaggleOps Architect\


---

## 📜 License

[MIT License](LICENSE)

