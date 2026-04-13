# End-to-End MLOps Pipeline on Google Cloud (2026)

This repository demonstrates a production-grade MLOps architecture using Google Cloud Platform. It moves beyond simple model training into automated CI/CD and Managed Orchestration.

## 🏗️ Architecture
- **Orchestration:** Vertex AI Pipelines (Kubeflow/KFP) for managed training logic and lineage tracking.
- **CI/CD:** GitHub + Cloud Build for automated containerization and deployment.
- **Serving Layer:** Cloud Run (Serverless API) providing a secure FastAPI/Flask endpoint.
- **Artifact Registry:** Google Cloud Storage (GCS) for decoupled model storage.



## 🚀 Key Features
- **Decoupled Weights:** The API fetches the model weight (`model.joblib`) from GCS at runtime, allowing model updates without redeploying code.
- **Identity-Aware Security:** The prediction endpoint is protected by IAM and requires a Google Identity Token.
- **Automated Lifecycle:** Every `git push` triggers a build, and every Pipeline run archives metadata for auditability.

## 🛠️ How to Run
1. **Train:** Run `vertex_pipeline.py` to trigger a Vertex AI Pipeline job.
2. **Deploy:** Push changes to GitHub to trigger the Cloud Build CI/CD.
3. **Predict:** Use the provided `curl` command with a Bearer token to get real-time inferences.

---
*Created by SUDULA SRI RAM REDDY | 2025 Graduate Freshers*
