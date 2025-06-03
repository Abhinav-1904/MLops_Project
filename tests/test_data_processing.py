import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    load_data,
    create_preprocessing_pipeline,
    prepare_data,
    save_preprocessor,
    load_preprocessor,
)
import os
import tempfile


@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    data = {
        "class": ["edible", "poisonous", "edible"],
        "cap-shape": ["convex", "flat", "convex"],
        "cap-surface": ["smooth", "scaly", "smooth"],
        "cap-color": ["brown", "red", "white"],
        "bruises": ["bruises", "no", "bruises"],
        "odor": ["none", "foul", "none"],
    }
    return pd.DataFrame(data)


def test_load_data(tmp_path):
    """Test loading data from a CSV file."""
    # Create a temporary CSV file
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)

    # Test loading the data
    loaded_df = load_data(file_path)
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == 3
    assert list(loaded_df.columns) == ["A", "B"]


def test_create_preprocessing_pipeline():
    """Test creating a preprocessing pipeline."""
    categorical_features = ["cap-shape", "cap-surface"]
    numerical_features = []

    preprocessor = create_preprocessing_pipeline(
        categorical_features, numerical_features
    )
    assert preprocessor is not None
    assert hasattr(preprocessor, "transform")


def test_prepare_data(sample_data):
    """Test preparing data for training."""
    X_train, X_test, y_train, y_test = prepare_data(
        sample_data, target_column="class", test_size=0.33, random_state=42
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # Check that target column is removed from features
    assert "class" not in X_train.columns
    assert "class" not in X_test.columns


def test_save_and_load_preprocessor():
    """Test saving and loading a preprocessor."""
    # Create a sample preprocessor
    categorical_features = ["cap-shape", "cap-surface"]
    numerical_features = []
    preprocessor = create_preprocessing_pipeline(
        categorical_features, numerical_features
    )

    # Save and load the preprocessor
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "preprocessor.pkl")
        save_preprocessor(preprocessor, filepath)
        loaded_preprocessor = load_preprocessor(filepath)

        assert loaded_preprocessor is not None
        assert hasattr(loaded_preprocessor, "transform")
