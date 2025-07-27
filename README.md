# MLOps End-to-End Pipeline


![Repository](https://img.shields.io/badge/Repo-mlops%20end2end%20pipeline-red)
![Dataset](https://img.shields.io/badge/Dataset-California%20Housing%20Dataset-Orange)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![scikit-learn](https://img.shields.io/badge/scikit-learn+-yellow)
![NumPy](https://img.shields.io/badge/NumPy-grey)
![Docker](https://img.shields.io/badge/Docker+-blue)
![GitHub Actions](https://img.shields.io/badge/GitHubAction-pink)
![ML](https://img.shields.io/badge/ML-Linear%20Regression-orange)

---

## Objective

Build an end-to-end MLOps pipeline that trains, quantizes, validates, and deploys a regression model using GitHub Actions CI/CD and publish to DockerHub.

---

## Project Structure

```
mlops-end2end-pipeline/
├── src/
│   ├── train.py                # Train linear regression model
│   ├── predict.py              # Predict 
│   ├── quantize.py             # Extract and quantize model weights and peform Inference using quantized weights
├── config/
│   └── config.json             # Configuration
├── requirements.txt
├── Dockerfile
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions CI pipeline
```

---

## Setup Instructions

```bash
# Step 1: Create and activate environment
conda create -n e2e_mlops_venv python=3.10 -y
conda activate e2e_mlops_venv

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Train the model
python src/train.py

# Step 4: Quantize the model weights & Perform inference
python src/quantize.py
```

---

## GitHub Actions Workflow (`ci.yml`)

This unified CI pipeline performs:

| Phase | Job | Description |
|-------|-----|-------------|
| Phase 1 | `train_model` | Trains model and saves `model.joblib` |
| Phase 3 | `quantize_model` | Quantizes weights into `quant_params.joblib` |
| Phase 2 | `docker_build_and_push` | Validates inference, builds image, and pushes to DockerHub |

Trigger branches: `main`, `docker_ci`, `quantization`

---

## Model Comparison: Sklearn vs Quantized

| **Metric**         | **Original Sklearn Model**         | **Quantized Model**            |
|--------------------|------------------------------------|---------------------------------|
| **R² Score**        | `0.5758`                           | `-333.4334`                     |
| **Model Size**      | `unquant_params.joblib` <br> `0.40 KB` | `quant_params.joblib` <br> `0.36 KB` |

---

## Analysis & Key Insight

- Quantization **reduced model size** slightly but severely impacted **model performance**.
- The drop in **R² score** highlights that the manually quantized weights deviate heavily from the original.
- This demonstrates the trade-off between **size** and **accuracy** in model deployment scenarios.

---


## Docker Deployment

- Docker builds an image using trained and quantized model.
- Entry point runs `quantize.py`.
- Image pushed to DockerHub on successful CI.

**DockerHub Secrets (set in GitHub > Settings > Secrets):**

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

---

## Output Sample

```bash
 Quantized weights saved to quant_params.joblib
 Inference with quantized weights complete.
 Predictions (first 5): [2.85 3.16 1.94 2.25 2.68]
```

---

## Key Insights

- Quantization reduces model size and enables lightweight inference.
- Modular Python scripts ensure separation of training, quantization, and inference.
- GitHub Actions automates testing, packaging, and Docker deployment.
- Docker guarantees reproducibility across environments.

---

##  Conclusion

This repo demonstrates how to operationalize an ML pipeline by combining modular code, version control, continuous integration, quantization, and containerized deployment. The result is a scalable and production-ready MLOps workflow.
