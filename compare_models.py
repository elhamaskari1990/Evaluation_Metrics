import csv
import math
import os
import random

from evaluation_metrics import EvaluationMetrics


MAX_SAMPLES = 500
EPOCHS = 10
SEED = 42


# ============================================================
# Text preprocessing
# ============================================================

def tokenize(text):
    text = text.lower()
    words = text.split()

    cleaned_words = []

    for word in words:
        word = word.strip(".,!?;:\"'()[]<>/\\")
        if word != "":
            cleaned_words.append(word)

    return cleaned_words


# ============================================================
# Data loading and train/test split
# ============================================================

def load_imdb(path, max_samples=MAX_SAMPLES):
    texts = []
    labels = []

    with open(path, encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for i, row in enumerate(reader):
            if i >= max_samples:
                break

            text = row["review"].strip()
            sentiment = row["sentiment"].strip().lower()

            if text and sentiment in ("positive", "negative"):
                texts.append(text)
                labels.append(sentiment)

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


def to_int(labels):
    result = []

    for label in labels:
        if label == "positive":
            result.append(1)
        else:
            result.append(0)

    return result


# ============================================================
# Model 1: Naive Bayes
# ============================================================

def nb_train(train_texts, train_labels):
    class_counts = {}
    word_counts = {}
    total_words = {}
    vocab = set()

    for text, label in zip(train_texts, train_labels):
        class_counts[label] = class_counts.get(label, 0) + 1

        if label not in word_counts:
            word_counts[label] = {}
            total_words[label] = 0

        words = tokenize(text)

        for word in words:
            vocab.add(word)
            word_counts[label][word] = word_counts[label].get(word, 0) + 1
            total_words[label] += 1

    total_docs = len(train_texts)
    priors = {}

    for label in class_counts:
        priors[label] = class_counts[label] / total_docs

    vocab_size = len(vocab)

    likelihoods = {}

    for label in word_counts:
        likelihoods[label] = {}

        for word in vocab:
            count = word_counts[label].get(word, 0)
            likelihoods[label][word] = (count + 1) / (total_words[label] + vocab_size)

    return priors, likelihoods, total_words, vocab_size


def nb_predict(text, priors, likelihoods, total_words, vocab_size):
    best_label = None
    best_score = None

    for label in priors:
        score = math.log(priors[label])

        for word in tokenize(text):
            if word in likelihoods[label]:
                score += math.log(likelihoods[label][word])
            else:
                score += math.log(1 / (total_words[label] + vocab_size))

        if best_score is None or score > best_score:
            best_score = score
            best_label = label

    return best_label


# ============================================================
# Model 2: Perceptron
# ============================================================

def perc_build_vocab(texts):
    vocab = {}

    for text in texts:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def perc_vectorize(text, vocab):
    vector = {}

    for word in tokenize(text):
        if word in vocab:
            index = vocab[word]
            vector[index] = 1

    return vector


def perc_train(train_texts, train_labels, epochs=EPOCHS):
    label_map = {
        "positive": 1,
        "negative": -1
    }

    vocab = perc_build_vocab(train_texts)

    weights = {}
    bias = 0

    def predict_vector(x):
        score = bias

        for index in x:
            score += weights.get(index, 0) * x[index]

        if score > 0:
            return 1
        return -1

    for epoch in range(epochs):
        errors = 0

        for text, label in zip(train_texts, train_labels):
            y = label_map[label]
            x = perc_vectorize(text, vocab)

            prediction = predict_vector(x)

            if prediction != y:
                for index in x:
                    weights[index] = weights.get(index, 0) + y

                bias += y
                errors += 1

        print("  Epoch", epoch + 1, "| errors =", errors)

    return vocab, weights, bias


def perc_predict(text, vocab, weights, bias):
    x = perc_vectorize(text, vocab)

    score = bias

    for index in x:
        score += weights.get(index, 0) * x[index]

    if score > 0:
        return "positive"
    return "negative"


# ============================================================
# Model 3: Logistic Regression
# ============================================================

def lr_build_vocab(texts):
    vocab = {}

    for text in texts:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def lr_vectorize(text, vocab):
    vector = {}

    for word in tokenize(text):
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


def lr_train(train_texts, train_labels, vocab, epochs=10, learning_rate=0.1):
    labels_int = to_int(train_labels)

    weights = {}
    bias = 0.0

    for epoch in range(epochs):
        total_error = 0.0

        for text, y in zip(train_texts, labels_int):
            x = lr_vectorize(text, vocab)

            score = bias

            for index in x:
                score += weights.get(index, 0.0) * x[index]

            probability = sigmoid(score)
            error = probability - y

            for index in x:
                weights[index] = weights.get(index, 0.0) - learning_rate * error * x[index]

            bias = bias - learning_rate * error
            total_error += abs(error)

        print("  Epoch", epoch + 1, "| total error =", round(total_error, 2))

    return weights, bias


def lr_predict(text, vocab, weights, bias):
    x = lr_vectorize(text, vocab)

    score = bias

    for index in x:
        score += weights.get(index, 0.0) * x[index]

    probability = sigmoid(score)

    if probability >= 0.5:
        return 1
    return 0


# ============================================================
# Result helpers
# ============================================================

def print_results(model_name, y_true, y_pred):
    evaluator = EvaluationMetrics(y_true, y_pred)

    tp, fp, fn, tn = evaluator.confusion_matrix()

    print("\n" + "━" * 44)
    print(" ", model_name)
    print("━" * 44)
    print("  Accuracy  :", round(evaluator.accuracy(), 4))
    print("  Precision :", round(evaluator.precision(), 4))
    print("  Recall    :", round(evaluator.recall(), 4))
    print("  F1 Score  :", round(evaluator.f1_score(), 4))
    print("  Confusion → TP =", tp, "FP =", fp, "FN =", fn, "TN =", tn)


def print_error_samples(model_name, test_texts, y_true, y_pred, n=3):
    label_map = {
        1: "positive",
        0: "negative"
    }

    errors = []

    for text, true, pred in zip(test_texts, y_true, y_pred):
        if true != pred:
            errors.append((text, true, pred))

    print("\n  --", model_name + ":", len(errors), "errors --")

    for text, true, pred in errors[:n]:
        print("  True:", label_map[true], "| Pred:", label_map[pred])
        print('  "' + text[:100] + '..."')


def print_top_words_perc(vocab, weights, n=10):
    index_to_word = {}

    for word, index in vocab.items():
        index_to_word[index] = word

    valid_weights = []

    for index, weight in weights.items():
        if index in index_to_word:
            valid_weights.append((index, weight))

    sorted_weights = sorted(valid_weights, key=lambda x: x[1], reverse=True)

    print("\n" + "─" * 44)
    print("  Perceptron — Top Word Weights")
    print("─" * 44)

    print("  POSITIVE signal words:")
    for index, weight in sorted_weights[:n]:
        print("   ", index_to_word[index], weight)

    print("  NEGATIVE signal words:")
    for index, weight in sorted_weights[-n:]:
        print("   ", index_to_word[index], weight)


def print_top_words_lr(vocab, weights, n=10):
    index_to_word = {}

    for word, index in vocab.items():
        index_to_word[index] = word

    valid_weights = []

    for index, weight in weights.items():
        if index in index_to_word:
            valid_weights.append((index, weight))

    sorted_weights = sorted(valid_weights, key=lambda x: x[1], reverse=True)

    print("\n" + "─" * 44)
    print("  Logistic Regression — Top Word Weights")
    print("─" * 44)

    print("  POSITIVE signal words:")
    for index, weight in sorted_weights[:n]:
        print("   ", index_to_word[index], round(weight, 4))

    print("  NEGATIVE signal words:")
    for index, weight in sorted_weights[-n:]:
        print("   ", index_to_word[index], round(weight, 4))


def cross_model_analysis(test_texts, y_true, nb_preds, perc_preds, lr_preds, n=4):
    label_map = {
        1: "positive",
        0: "negative"
    }

    case1 = []
    case2 = []
    case3 = []

    for text, true, nb, perc, lr in zip(test_texts, y_true, nb_preds, perc_preds, lr_preds):
        if true != nb and true != perc and true == lr:
            case1.append((text, true, nb, perc, lr))

        if true != nb and true != perc and true != lr:
            case2.append((text, true, nb, perc, lr))

        if true == nb and true != perc and true != lr:
            case3.append((text, true, nb, perc, lr))

    print("\n\n" + "═" * 60)
    print("   CROSS-MODEL ERROR ANALYSIS")
    print("═" * 60)

    print("\n[CASE 1] NB wrong + Perceptron wrong + LR correct:", len(case1), "reviews")
    for text, true, nb, perc, lr in case1[:n]:
        print("\n  True =", label_map[true],
              "| NB =", label_map[nb],
              "| Perc =", label_map[perc],
              "| LR =", label_map[lr])
        print('  Review: "' + text[:220] + '"')

    print("\n[CASE 2] All three models wrong:", len(case2), "reviews")
    for text, true, nb, perc, lr in case2[:n]:
        print("\n  True =", label_map[true],
              "| NB =", label_map[nb],
              "| Perc =", label_map[perc],
              "| LR =", label_map[lr])
        print('  Review: "' + text[:220] + '"')

    print("\n[CASE 3] Only NB correct:", len(case3), "reviews")
    for text, true, nb, perc, lr in case3[:n]:
        print("\n  True =", label_map[true],
              "| NB =", label_map[nb],
              "| Perc =", label_map[perc],
              "| LR =", label_map[lr])
        print('  Review: "' + text[:220] + '"')


# ============================================================
# Main
# ============================================================

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    dataset_file = os.path.join(base, "IMDB Dataset.csv")

    print("Loading", MAX_SAMPLES, "samples from IMDB dataset...")
    texts, labels = load_imdb(dataset_file)

    print("Splitting 80/20 with seed =", SEED)
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels)

    print("Train:", len(train_texts), "| Test:", len(test_texts))

    test_labels_int = to_int(test_labels)

    print("\n[1/3] Training Naive Bayes...")
    priors, likelihoods, total_words, vocab_size = nb_train(train_texts, train_labels)
    nb_preds = []

    for text in test_texts:
        nb_preds.append(nb_predict(text, priors, likelihoods, total_words, vocab_size))

    nb_preds_int = to_int(nb_preds)

    print("\n[2/3] Training Perceptron...")
    perc_vocab, perc_weights, perc_bias = perc_train(train_texts, train_labels)
    perc_preds = []

    for text in test_texts:
        perc_preds.append(perc_predict(text, perc_vocab, perc_weights, perc_bias))

    perc_preds_int = to_int(perc_preds)

    print("\n[3/3] Training Logistic Regression...")
    lr_vocab = lr_build_vocab(train_texts)
    lr_weights, lr_bias = lr_train(train_texts, train_labels, lr_vocab, epochs=10, learning_rate=0.1)
    lr_preds = []

    for text in test_texts:
        lr_preds.append(lr_predict(text, lr_vocab, lr_weights, lr_bias))

    print("\n\n" + "═" * 44)
    print("   MODEL COMPARISON")
    print("═" * 44)

    print_results("Naive Bayes", test_labels_int, nb_preds_int)
    print_results("Perceptron", test_labels_int, perc_preds_int)
    print_results("Logistic Regression", test_labels_int, lr_preds)

    print("\n\n" + "═" * 44)
    print("       ERROR ANALYSIS SAMPLES")
    print("═" * 44)

    print_error_samples("Naive Bayes", test_texts, test_labels_int, nb_preds_int)
    print_error_samples("Perceptron", test_texts, test_labels_int, perc_preds_int)
    print_error_samples("Logistic Regression", test_texts, test_labels_int, lr_preds)

    print("\n\n" + "═" * 44)
    print("   WORD WEIGHT ANALYSIS")
    print("═" * 44)

    print_top_words_perc(perc_vocab, perc_weights)
    print_top_words_lr(lr_vocab, lr_weights)

    cross_model_analysis(test_texts, test_labels_int, nb_preds_int, perc_preds_int, lr_preds)


if __name__ == "__main__":
    main()