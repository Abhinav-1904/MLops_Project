import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_data(file_path):
    """
    Load the mushroom dataset from the specified path.
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(file_path)

def create_preprocessing_pipeline(categorical_features, numerical_features):
    """
    Create a preprocessing pipeline for the dataset.
    
    Args:
        categorical_features (list): List of categorical feature names
        numerical_features (list): List of numerical feature names
        
    Returns:
        sklearn.pipeline.Pipeline: Preprocessing pipeline
    """
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features)
        ])
    
    return preprocessor

def prepare_data(df, target_column, test_size=0.2, random_state=42):
    """
    Prepare the dataset for training by splitting features and target,
    and creating train-test splits.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_preprocessor(preprocessor, filepath):
    """
    Save the preprocessor to disk.
    
    Args:
        preprocessor: The preprocessor object to save
        filepath (str): Path where to save the preprocessor
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)

def load_preprocessor(filepath):
    """
    Load a preprocessor from disk.
    
    Args:
        filepath (str): Path to the saved preprocessor
        
    Returns:
        The loaded preprocessor object
    """
    return joblib.load(filepath) 