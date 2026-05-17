import csv
import math
import os
import random

from evaluation_metrics import EvaluationMetrics


MAX_SAMPLES = 500
EPOCHS = 10
LEARNING_RATE = 0.1
SEED = 42


def tokenize(text):
    text = text.lower()
    words = text.split()

    cleaned_words = []

    for word in words:
        word = word.strip(".,!?;:\"'()[]<>/\\")
        if word != "":
            cleaned_words.append(word)

    return cleaned_words


def load_imdb(filename, max_samples=MAX_SAMPLES):
    texts = []
    labels = []

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for i, row in enumerate(reader):
            if i >= max_samples:
                break

            text = row["review"].strip()
            sentiment = row["sentiment"].strip().lower()

            if text and sentiment in ("positive", "negative"):
                texts.append(text)

                if sentiment == "positive":
                    labels.append(1)
                else:
                    labels.append(0)

    return texts, labels


def train_test_split(texts, labels, test_size=0.2, seed=SEED):
    combined = list(zip(texts, labels))

    random.seed(seed)
    random.shuffle(combined)

    split_index = int(len(combined) * (1 - test_size))

    train = combined[:split_index]
    test = combined[split_index:]

    train_texts, train_labels = zip(*train)
    test_texts, test_labels = zip(*test)

    return list(train_texts), list(test_texts), list(train_labels), list(test_labels)


def build_vocab(data):
    vocab = {}

    for text, label in data:
        words = tokenize(text)

        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def vectorize(text, vocab):
    vector = {}

    words = tokenize(text)

    for word in words:
        if word in vocab:
            index = vocab[word]
            vector[index] = 1

    return vector


def sigmoid(z):
    if z < -500:
        return 0.0
    if z > 500:
        return 1.0

    return 1 / (1 + math.exp(-z))


def predict_probability(x, weights, bias):
    score = bias

    for index in x:
        score += weights.get(index, 0.0) * x[index]

    return sigmoid(score)


def predict(x, weights, bias):
    probability = predict_probability(x, weights, bias)

    if probability >= 0.5:
        return 1
    return 0


def train(train_texts, train_labels, vocab, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    weights = {}
    bias = 0.0

    for epoch in range(epochs):
        total_loss = 0.0

        for text, true_label in zip(train_texts, train_labels):
            x = vectorize(text, vocab)

            probability = predict_probability(x, weights, bias)
            error = probability - true_label

            p_safe = max(1e-10, min(1 - 1e-10, probability))
            loss = -(true_label * math.log(p_safe) + (1 - true_label) * math.log(1 - p_safe))
            total_loss += loss

            for index in x:
                weights[index] = weights.get(index, 0.0) - learning_rate * error * x[index]

            bias = bias - learning_rate * error

        average_loss = total_loss / len(train_texts)
        print("  Epoch", epoch + 1, "| loss =", round(average_loss, 4))

    return weights, bias


def evaluate(test_texts, test_labels, vocab, weights, bias):
    predictions = []

    for text in test_texts:
        x = vectorize(text, vocab)
        predictions.append(predict(x, weights, bias))

    evaluator = EvaluationMetrics(test_labels, predictions)

    tp, fp, fn, tn = evaluator.confusion_matrix()

    print("\n===== Results =====")
    print("Accuracy  :", round(evaluator.accuracy(), 4))
    print("Precision :", round(evaluator.precision(), 4))
    print("Recall    :", round(evaluator.recall(), 4))
    print("F1 Score  :", round(evaluator.f1_score(), 4))
    print("Confusion Matrix:")
    print("TP =", tp, "FP =", fp, "FN =", fn, "TN =", tn)

    print("\n===== Error Analysis =====")

    label_map = {
        1: "positive",
        0: "negative"
    }

    errors = []

    for text, true_label, predicted_label in zip(test_texts, test_labels, predictions):
        if true_label != predicted_label:
            errors.append((text, true_label, predicted_label))

    print("Total errors:", len(errors), "/", len(test_labels))

    for text, true_label, predicted_label in errors[:5]:
        print("\nTrue:", label_map[true_label], "| Predicted:", label_map[predicted_label])
        print('"' + text[:110] + '..."')

    return predictions


def top_words(vocab, weights, n=10):
    index_to_word = {}

    for word, index in vocab.items():
        index_to_word[index] = word

    valid_weights = []

    for index, weight in weights.items():
        if index in index_to_word:
            valid_weights.append((index, weight))

    sorted_weights = sorted(valid_weights, key=lambda x: x[1], reverse=True)

    print("\n===== Top Word Weights: Logistic Regression =====")

    print("\nPositive signal words:")
    for index, weight in sorted_weights[:n]:
        print(index_to_word[index], "=", round(weight, 4))

    print("\nNegative signal words:")
    for index, weight in sorted_weights[-n:]:
        print(index_to_word[index], "=", round(weight, 4))


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base, "IMDB Dataset.csv")

    print("Loading dataset...")
    texts, labels = load_imdb(dataset_file, MAX_SAMPLES)

    print("Splitting 80/20 with seed =", SEED)
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels)

    print("Train:", len(train_texts), "| Test:", len(test_texts))

    print("\nBuilding vocabulary...")
    vocab = build_vocab(list(zip(train_texts, train_labels)))
    print("Vocabulary size:", len(vocab))

    print("\nTraining Logistic Regression...")
    weights, bias = train(
        train_texts,
        train_labels,
        vocab,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )

    evaluate(test_texts, test_labels, vocab, weights, bias)

    top_words(vocab, weights)


if __name__ == "__main__":
    main()