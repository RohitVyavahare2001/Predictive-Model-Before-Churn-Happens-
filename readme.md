```markdown
# Customer Churn Prediction System

## Overview
This project builds a predictive model that identifies customers at risk of churning before they show obvious signs. The model is designed to work in production and explain its decisions to help the retention team take action.

### Key Features:
- Predicts churn probability for each customer before they show clear churn signals.
- Explains predictions with feature importance to help decision-makers.
- Works in production using a FastAPI backend and a Streamlit dashboard.
- Tracks customers with a unique `customer_id`.
- Visualizes insights using interactive charts.

## Project Structure
```
📂 churn_prediction_project
 ├── churn_prediction_pipeline.py  # Data processing & model training
 ├── churn_prediction_api.py        # FastAPI backend for predictions
 ├── dashboard_app.py               # Streamlit dashboard
 ├── churn_prediction_model.joblib   # Trained model
 ├── requirements.txt                # Dependencies
 ├── README.md                       # Project Documentation
```

## How to Run Locally
Follow these steps to set up and run the project.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python churn_prediction_pipeline.py
```
This will generate synthetic customer data, train a model, and save it as `churn_prediction_model.joblib`.

### 3. Start the API Server
```bash
uvicorn churn_prediction_api:app --reload
```
The API will be available at: [http://localhost:8000](http://localhost:8000)  
Test the API in the browser: [http://localhost:8000/docs](http://localhost:8000/docs)  

### 4. Run the Dashboard
```bash
streamlit run dashboard_app.py
```
The dashboard will open in your browser, allowing you to visualize predictions and business insights.

## Deploying to Production
This system is production-ready and can be deployed easily. Here’s how:

### Option 1: Deploy on a Cloud Server (AWS, GCP, Azure)
1. Deploy API using FastAPI + Uvicorn
```bash
uvicorn churn_prediction_api:app --host 0.0.0.0 --port 8000
```
2. Host the Streamlit Dashboard
```bash
streamlit run dashboard_app.py --server.port 8501 --server.address 0.0.0.0
```

### Option 2: Deploy with Docker
1. Build the Docker Image:
```bash
docker build -t churn-prediction .
```
2. Run the API Container:
```bash
docker run -p 8000:8000 churn-prediction
```
3. Run the Dashboard Container:
```bash
docker run -p 8501:8501 churn-prediction-dashboard
```

### Option 3: Use a Managed Service
- Deploy the API to Google Cloud Run, AWS Lambda, or Azure App Services.
- Deploy the dashboard using Streamlit Cloud or a cloud VM.

## Business Value
This system helps businesses reduce churn by:
- Identifying at-risk customers early.
- Providing actionable insights for the retention team.
- Helping prioritize high-value customers for retention efforts.

---
Now the project is fully functional and can be deployed easily for real-world use.
```

