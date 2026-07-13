# IMDb Sentiment Classification with BERT

## Overview

This project fine-tunes a pretrained BERT model to classify IMDb movie reviews as positive or negative.

The main purpose of the project is to apply transfer learning and understand the fine-tuning process of Transformer-based language models for a text classification task.

## Features

- IMDb movie review sentiment classification
- Binary classification: positive and negative
- BERT tokenization
- Pretrained BERT fine-tuning
- Training and validation pipeline
- Model performance evaluation
- Prediction on custom movie reviews

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- BERT
- Pandas
- Scikit-learn

## Workflow

```text
IMDb Reviews
      ↓
Text Preprocessing
      ↓
BERT Tokenizer
      ↓
Pretrained BERT Model
      ↓
Fine-tuning
      ↓
Positive / Negative Prediction
```

## Model

A pretrained BERT model was adapted for binary sequence classification. The model receives tokenized movie reviews and predicts whether each review expresses positive or negative sentiment.

## Evaluation Metrics

The model can be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python train.py
```

Run predictions:

```bash
python predict.py
```

> Update the commands above according to the actual filenames in the repository.

## Future Improvements

- Hyperparameter optimization
- Comparison with LSTM and traditional machine learning models
- Testing different pretrained Transformer models
- Deployment with a simple web interface

## Author

**Abdullah Seha**