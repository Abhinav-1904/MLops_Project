# Mushroom Classification Project

This project implements a multinomial classification system for mushroom classification using the UCI Mushroom Dataset. The system can classify mushrooms into multiple categories based on their characteristics.

## Dataset

The project uses the UCI Mushroom Dataset, which contains various attributes of mushrooms and their classification. The dataset is available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom).

## Project Structure

```
.
├── app/
│   └── app.py              # Streamlit web application
├── data/
│   ├── raw/               # Raw dataset
│   └── processed/         # Processed dataset
├── models/                # Trained models and preprocessors
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── data_processing.py # Data processing pipeline
│   └── model.py          # Model training and evaluation
├── tests/                # Unit tests
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd mushroom-classification
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app/app.py
```

## Data Generation and Model Training

1. Download and process the dataset:
```bash
python src/data_processing.py
```

2. Train the model:
```bash
python src/model.py
```

The data processing script will:
- Download the UCI Mushroom dataset
- Clean and preprocess the data
- Save processed data to `data/processed/`

The model training script will:
- Load the processed data
- Train the classification model
- Save the trained model and preprocessor to `models/`

## Docker Support

Build and run using Docker:
```bash
docker build -t mushroom-classification .
docker run -p 8501:8501 mushroom-classification
```

## Development

- Code style: Black
- Linting: Flake8
- Testing: Pytest

## Future Enhancements

- [ ] Add model monitoring and logging
- [ ] Implement A/B testing framework
- [ ] Add more advanced feature engineering
- [ ] Deploy to cloud platform
- [ ] Add API documentation
- [ ] Implement model versioning

## License

MIT License 