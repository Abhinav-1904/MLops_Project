import pandas as pd
import numpy as np
import os
import requests
from pathlib import Path


def download_mushroom_dataset():
    """
    Download the UCI Mushroom dataset and save it to the data/raw directory.
    """
    # Create data/raw directory if it doesn't exist
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # URL for the UCI Mushroom dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

    # Column names for the dataset
    columns = [
        "class",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ]

    try:
        # Download the dataset
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Save the raw data
        raw_data_path = raw_data_dir / "mushroom_data.csv"
        with open(raw_data_path, "w") as f:
            f.write(response.text)

        # Read and process the data
        df = pd.read_csv(raw_data_path, names=columns)

        # Save the processed data
        processed_data_dir = Path("data/processed")
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        processed_data_path = processed_data_dir / "mushroom_data_processed.csv"
        df.to_csv(processed_data_path, index=False)

        print(f"Dataset downloaded and saved to {raw_data_path}")
        print(f"Processed dataset saved to {processed_data_path}")
        print("\nDataset Info:")
        print(f"Number of samples: {len(df)}")
        print(f"Number of features: {len(df.columns) - 1}")  # Excluding target
        print("\nClass distribution:")
        print(df["class"].value_counts())

        return df

    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        return None


if __name__ == "__main__":
    download_mushroom_dataset()
