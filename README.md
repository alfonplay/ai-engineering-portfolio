# AI Engineering Portfolio — Alfonso Lema

A series of end-to-end machine learning and data engineering projects developed during the **AnyoneAI AI Engineering Program**.

Each sprint builds on the previous one, progressing from data pipelines to production-ready ML systems.

---

## Projects

### [Sprint 01 — E-Commerce Data Pipeline](./sprint-01-ecommerce-pipeline/)
> **ELT Pipeline · SQL · Pandas · Data Analysis**

Built a full Extract-Load-Transform (ELT) pipeline over 100k+ real e-commerce orders (Olist, Brazil 2016–2018). Includes 9 SQL analytical queries covering revenue trends, delivery performance, and product category analysis. Data enriched with a public holidays API.

**Stack:** Python · Pandas · SQLite · SQLAlchemy · SQL · Matplotlib · Seaborn · pytest

---

### [Sprint 02 — Credit Risk Prediction](./sprint-02-credit-risk-ml/)
> **Binary Classification · Feature Engineering · scikit-learn**

Predicted whether a home credit applicant will have payment difficulties. Built a complete preprocessing pipeline: outlier correction, categorical encoding (Ordinal + One-Hot), median imputation, and Min-Max scaling. Evaluated using AUC-ROC.

**Stack:** Python · Pandas · scikit-learn · Matplotlib · Seaborn · Jupyter

---

### [Sprint 03 — Multimodal Product Classification](./sprint-03-multimodal-classification/)
> **Transfer Learning · Embeddings · Deep Learning · Multimodal AI**

Classified BestBuy products into categories using both product images and text descriptions. Extracted embeddings using pre-trained vision models (ResNet50, ConvNextV2, ViT, Swin Transformer) and language models (MiniLM, BERT). Trained both classic ML classifiers and MLP models on the fused embeddings.

- Multimodal model: **≥85% accuracy, ≥80% F1-score**
- Text-only model: **≥85% accuracy, ≥80% F1-score**
- Image-only model: **≥75% accuracy, ≥70% F1-score**

**Stack:** Python · TensorFlow · Hugging Face Transformers · scikit-learn · Plotly · Pandas

---

### [Sprint 04 — Image Classification API (Production)](./sprint-04-fastapi-ml-app/)
> **MLOps · Microservices · FastAPI · Docker · Redis**

Deployed a full production ML system for classifying images into 1000+ ImageNet categories. Implemented an async inference architecture using Redis as a job queue between a FastAPI REST service and a TensorFlow ResNet50 model server. Includes JWT authentication, user management, feedback collection, and a Streamlit web UI.

**Architecture:**
```
[Streamlit UI] → [FastAPI + JWT + PostgreSQL] → [Redis Queue] → [ResNet50 ML Service]
```

**Stack:** Python · FastAPI · TensorFlow · Redis · Streamlit · Docker · docker-compose · SQLAlchemy · PostgreSQL · Locust · pytest

---

## Skills Demonstrated

| Area | Technologies |
|------|-------------|
| Data Engineering | Python, Pandas, SQL, SQLite, SQLAlchemy, ELT pipelines |
| Machine Learning | scikit-learn, feature engineering, classification, AUC-ROC |
| Deep Learning | TensorFlow, Keras, transfer learning, embeddings |
| NLP | Hugging Face Transformers, sentence-transformers, BERT |
| MLOps & Production | FastAPI, Docker, Redis, async inference, JWT, PostgreSQL |
| Testing | pytest, unit tests, integration tests, stress testing (Locust) |

---

## Program

**AnyoneAI — AI Engineering Program**
