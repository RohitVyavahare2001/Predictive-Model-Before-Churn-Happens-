import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import shap
import joblib

class ChurnPredictionPipeline:
    def __init__(self, model_name="churn_prediction_model"):
        self.model_name = model_name
        self.model_pipeline = None
        self.numerical_features = [
            'tenure', 'monthly_charges', 'total_charges', 'avg_session_length',
            'avg_session_count_monthly', 'days_since_last_activity',
            'support_tickets_count', 'items_purchased_6m', 'avg_purchase_value'
        ]
        self.categorical_features = [
            'contract_type', 'payment_method', 'subscription_tier',
            'has_premium_support', 'has_family_plan', 'has_multiple_devices'
        ]
        self.explainer = None
    
    def preprocess_data(self):
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(n_estimators=200, random_state=42))
        ])
        
    def train(self, X, y):
        if self.model_pipeline is None:
            self.preprocess_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model_pipeline.fit(X_train, y_train)
        
        self.explainer = shap.Explainer(self.model_pipeline.named_steps['classifier'])
        joblib.dump(self.model_pipeline, f"{self.model_name}.joblib")
        
    def predict(self, X):
        if self.model_pipeline is None:
            raise ValueError("Model not trained yet.")
        y_proba = self.model_pipeline.predict_proba(X)[:, 1]
        return y_proba
    
    def load_model(self, model_path):
        self.model_pipeline = joblib.load(model_path)
        self.explainer = shap.Explainer(self.model_pipeline.named_steps['classifier'])
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = f"{self.model_name}.joblib"
        joblib.dump(self.model_pipeline, model_path)
    
    def generate_synthetic_data(self, n_samples=1000):
        np.random.seed(42)
        data = {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n_samples + 1)],
            "tenure": np.random.gamma(2.0, 10.0, n_samples),
            "monthly_charges": np.random.normal(65, 30, n_samples),
            "total_charges": np.random.normal(1000, 500, n_samples),
            "avg_session_length": np.random.normal(25, 10, n_samples),
            "avg_session_count_monthly": np.random.poisson(15, n_samples),
            "days_since_last_activity": np.random.exponential(7, n_samples),
            "support_tickets_count": np.random.poisson(1, n_samples),
            "items_purchased_6m": np.random.poisson(3, n_samples),
            "avg_purchase_value": np.random.gamma(2.0, 20.0, n_samples),
            "contract_type": np.random.choice(["Monthly", "One-year", "Two-year"], n_samples, p=[0.6, 0.3, 0.1]),
            "payment_method": np.random.choice(["Credit card", "Bank transfer", "Electronic check", "Mailed check"], n_samples),
            "subscription_tier": np.random.choice(["Basic", "Standard", "Premium"], n_samples, p=[0.3, 0.5, 0.2]),
            "has_premium_support": np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
            "has_family_plan": np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
            "has_multiple_devices": np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
        }
        df = pd.DataFrame(data)
        df["churn"] = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        return df
    
if __name__ == "__main__":
    pipeline = ChurnPredictionPipeline()
    df = pipeline.generate_synthetic_data()
    X = df.drop(columns=["churn"])
    y = df["churn"]
    pipeline.train(X, y)
    print("Model trained and saved successfully.")
