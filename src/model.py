import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_model(X_train, y_train, model_type='logistic', **kwargs):
    """
    Train a classification model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_type (str): Type of model to train ('logistic' or 'random_forest')
        **kwargs: Additional parameters for the model
        
    Returns:
        Trained model
    """
    if model_type == 'logistic':
        model = LogisticRegression(multi_class='multinomial', max_iter=1000, **kwargs)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    metrics = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    return metrics

def save_model(model, filepath):
    """
    Save the trained model to disk.
    
    Args:
        model: The model to save
        filepath (str): Path where to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        The loaded model
    """
    return joblib.load(filepath)

def predict(model, X):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        X: Features to predict on
        
    Returns:
        tuple: (predictions, probabilities)
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return predictions, probabilities 