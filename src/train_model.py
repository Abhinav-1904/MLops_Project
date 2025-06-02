import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import (
    load_data,
    create_preprocessing_pipeline,
    prepare_data,
    save_preprocessor
)
from src.model import train_model, evaluate_model, save_model

def train_and_save_model():
    """
    Train the model on the UCI Mushroom dataset and save it along with the preprocessor.
    """
    try:
        # Load the dataset
        data_path = Path("data/processed/mushroom_data_processed.csv")
        print(f"Looking for data at: {data_path.absolute()}")
        
        if not data_path.exists():
            print("Processed data not found. Please run download_data.py first.")
            return
        
        print("Loading data...")
        df = load_data(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Define features
        categorical_features = [
            'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring',
            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
            'ring-type', 'spore-print-color', 'population', 'habitat'
        ]
        numerical_features = []  # No numerical features in this dataset
        
        # Create and fit preprocessor
        print("Creating preprocessor...")
        preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
        
        # Prepare data
        print("Preparing data...")
        X_train, X_test, y_train, y_test = prepare_data(
            df,
            target_column='class',
            test_size=0.2,
            random_state=42
        )
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Fit preprocessor
        print("Fitting preprocessor...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        print(f"Processed training set shape: {X_train_processed.shape}")
        
        # Train model
        print("\nTraining model...")
        model = train_model(
            X_train_processed,
            y_train,
            model_type='random_forest',
            n_estimators=100,
            random_state=42
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = evaluate_model(model, X_test_processed, y_test)
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model and preprocessor
        print("\nSaving model and preprocessor...")
        model_path = models_dir / "model.pkl"
        preprocessor_path = models_dir / "preprocessor.pkl"
        
        save_model(model, str(model_path))
        save_preprocessor(preprocessor, str(preprocessor_path))
        
        print(f"Model saved to: {model_path.absolute()}")
        print(f"Preprocessor saved to: {preprocessor_path.absolute()}")
        print("Model and preprocessor saved successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_and_save_model() 