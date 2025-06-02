import pytest
import numpy as np
import pandas as pd
from src.model import (
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict
)
import os
import tempfile

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1]
    })
    y = pd.Series(['A', 'B', 'A', 'B', 'A'])
    return X, y

def test_train_model(sample_data):
    """Test model training."""
    X, y = sample_data
    
    # Test logistic regression
    model = train_model(X, y, model_type='logistic')
    assert model is not None
    assert hasattr(model, 'predict')
    
    # Test random forest
    model = train_model(X, y, model_type='random_forest')
    assert model is not None
    assert hasattr(model, 'predict')
    
    # Test invalid model type
    with pytest.raises(ValueError):
        train_model(X, y, model_type='invalid')

def test_evaluate_model(sample_data):
    """Test model evaluation."""
    X, y = sample_data
    model = train_model(X, y, model_type='logistic')
    
    metrics = evaluate_model(model, X, y)
    
    assert isinstance(metrics, dict)
    assert 'classification_report' in metrics
    assert 'confusion_matrix' in metrics
    assert 'predictions' in metrics
    assert 'probabilities' in metrics

def test_save_and_load_model(sample_data):
    """Test saving and loading a model."""
    X, y = sample_data
    model = train_model(X, y, model_type='logistic')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'model.pkl')
        save_model(model, filepath)
        loaded_model = load_model(filepath)
        
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
        # Test that predictions are the same
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        np.testing.assert_array_equal(original_pred, loaded_pred)

def test_predict(sample_data):
    """Test making predictions."""
    X, y = sample_data
    model = train_model(X, y, model_type='logistic')
    
    predictions, probabilities = predict(model, X)
    
    assert len(predictions) == len(X)
    assert probabilities.shape == (len(X), len(model.classes_))
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0) 