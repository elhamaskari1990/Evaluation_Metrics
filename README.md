# Text Classification Project (Naive Bayes + Evaluation Metrics)

This project implements a simple text classification system from scratch using Python, without any external machine learning libraries.

## Part 1: Evaluation Metrics
Implemented from scratch:
- Confusion Matrix (TP, FP, FN, TN)
- Accuracy
- Precision
- Recall
- F1 Score
- Multi-class support

## Part 2: Naive Bayes Classifier
A Naive Bayes classifier was implemented for sentiment analysis.

### Features
- Tokenization of text
- Vocabulary building
- Word frequency counting
- Prior probabilities P(y)
- Likelihood probabilities P(word | class) with Laplace smoothing
- Text classification using:
  
  score(x, c) = P(c) * Π P(w | c)

### Dataset
IMDB movie reviews dataset (positive / negative)

### Process
1. Load dataset from CSV file
2. Split into training and test data
3. Train Naive Bayes model
4. Predict sentiment for test data
5. Evaluate predictions using evaluation metrics

## Results
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

## Notes
- No external libraries were used
- Everything is implemented from scratch
- The model uses word frequencies and does not understand context or grammar

## Example Predictions
- "this movie was amazing" → positive  
- "this movie was terrible" → negative