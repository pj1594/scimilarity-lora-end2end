
# SCimilarity + LoRA Fine-Tuning & Real-Time Deployment

**Author:** Prajwal Eachempati  
**Role:** AI & Process Automation Consultant | PhD   
**Focus:** LoRA-based fine-tuning, FastAPI inference, and Hugging Face deployment  

---

## Executive Summary

This project demonstrates **end-to-end AI deployment** for *single-cell transcriptomics classification* using:

- **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning  
- **FastAPI** backend for scalable inference  
- **ngrok** tunneling for public API exposure  
- **Streamlit + Hugging Face Spaces** for live, interactive prediction UI  

The result:  
> A fully functional, cloud-accessible single-cell type classifier achieving higher validation accuracy with just **2.3% of model parameters fine-tuned**.

---

## Core Achievements

| Area | Details |
|------|----------|
| **Base Model** | SCimilarity v1.1 (CZI Science) |
| **Adapter** | LoRA (r=8, rank-reduced) |
| **Accuracy Gain** | +50% |
| **Inference Speed** | 95 ms/sample |
| **Deployment** | Hugging Face Spaces + ngrok API |
| **Frontend** | Streamlit UI for cell-type prediction |
| **Backend** | FastAPI + PyTorch + SCimilarity Encoder |
| **DevOps Workflow** | Google Colab → GitHub → Hugging Face CI/CD |

---

## Model Workflow

1. **Load pre-trained SCimilarity encoder**
2. **Attach LoRA adapters** to target layers (fc1, fc2)
3. **Train linear head** on labeled single-cell data
4. **Evaluate model accuracy & macro-F1**
5. **Serve model with FastAPI**
6. **Expose public endpoint via ngrok**
7. **Connect Streamlit UI (Hugging Face)** to backend API

---

## Live Demo

**Frontend (Streamlit UI):**  
🔗 [https://huggingface.co/spaces/praj-1594/scimilarity-lora-ui](https://huggingface.co/spaces/praj-1594/scimilarity-lora-ui)

**Backend (FastAPI via ngrok):**  
🔗 [https://nasir-spacious-kamila.ngrok-free.dev/predict]

**Test Command:**
```bash
curl -X POST https://nasir-spacious-kamila.ngrok-free.dev/predict \
  -H "Content-Type: application/json" \
  -d '{"expression": {"CD19":8.5,"MS4A1":9.2,"CD79A":7.8}}'

Implementation methodology details discussed in README_documentation.md

## Quickstart
See `deploy/deploy_huggingface.sh` for  Hugging Face deployment once your artifacts are in place.
