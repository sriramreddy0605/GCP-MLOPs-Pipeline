# GCP MLOps Pipeline: Serverless Model Experimentation

This repository contains a complete MLOps workflow for containerized machine learning experiments on Google Cloud Platform.

## 🏗️ Project Components
- **experiment.py**: ML logic comparing CountVectorizer and TF-IDF strategies.
- **Dockerfile**: Containerization recipe for environment reproducibility.
- **GCP Cloud Run Jobs**: Serverless orchestration for batch training.
- **Artifact Registry**: Versioned storage for Docker images.

## 🛠️ Tech Stack
- **Languages**: Python 3.11
- **Tools**: Docker, Git, GCP SDK
- **ML Libraries**: Scikit-Learn, Pandas, Joblib

## 🚀 Deployment
The model is containerized and deployed as a **Cloud Run Job**, allowing for cost-effective, on-demand execution without managing virtual machines.
