import csv
from evaluation_metrics import EvaluationMetrics


def tokenize(text):
    text = text.lower()
    words = text.split()

    cleaned_words = []

    for word in words:
        word = word.strip(".,!?;:\"'()[]<>/\\")
        if word != "":
            cleaned_words.append(word)

    return cleaned_words


def load_imdb_dataset(filename, limit=200):
    dataset = []

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            text = row["review"]
            label = row["sentiment"]
            dataset.append((text, label))

            if len(dataset) == limit:
                break

    return dataset


def build_vocab(dataset):
    vocab = set()

    for text, label in dataset:
        words = tokenize(text)

        for word in words:
            vocab.add(word)

    return vocab


class NaiveBayesClassifier:

    def __init__(self):
        self.vocab = set()
        self.total_words = {}
        self.class_counts = {}
        self.word_counts = {}
        self.priors = {}
        self.likelihoods = {}
        self.labels = []

    def train(self, dataset):
        self.vocab = build_vocab(dataset)

        for text, label in dataset:
            if label not in self.labels:
                self.labels.append(label)
                self.total_words[label] = 0
                self.class_counts[label] = 0
                self.word_counts[label] = {}
                self.likelihoods[label] = {}

        for text, label in dataset:
            self.class_counts[label] += 1
            words = tokenize(text)

            for word in words:
                self.total_words[label] += 1

                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 1
                else:
                    self.word_counts[label][word] += 1

        total_samples = len(dataset)

        for label in self.labels:
            self.priors[label] = self.class_counts[label] / total_samples

        vocab_size = len(self.vocab)

        for label in self.labels:
            for word in self.vocab:
                count = 0

                if word in self.word_counts[label]:
                    count = self.word_counts[label][word]

                self.likelihoods[label][word] = (count + 1) / (self.total_words[label] + vocab_size)

    def score(self, text, label):
        words = tokenize(text)
        result = self.priors[label]
        vocab_size = len(self.vocab)

        for word in words:
            if word in self.likelihoods[label]:
                result = result * self.likelihoods[label][word]
            else:
                result = result * (1 / (self.total_words[label] + vocab_size))

        return result

    def predict(self, text):
        best_label = None
        best_score = None

        for label in self.labels:
            current_score = self.score(text, label)

            if best_score is None or current_score > best_score:
                best_score = current_score
                best_label = label

        return best_label


dataset = load_imdb_dataset("IMDB Dataset.csv", 200)

train_data = dataset[:160]
test_data = dataset[160:]

model = NaiveBayesClassifier()
model.train(train_data)

y_true_text = []
y_pred_text = []

for text, label in test_data:
    prediction = model.predict(text)

    y_true_text.append(label)
    y_pred_text.append(prediction)

print("Predictions:")
for i in range(len(test_data)):
    print("True:", y_true_text[i], "| Predicted:", y_pred_text[i])

print("\nEvaluation:")

# Convert labels to binary numbers for using last week's evaluation code
# positive = 1, negative = 0
y_true = []
y_pred = []

for label in y_true_text:
    if label == "positive":
        y_true.append(1)
    else:
        y_true.append(0)

for label in y_pred_text:
    if label == "positive":
        y_pred.append(1)
    else:
        y_pred.append(0)

evaluator = EvaluationMetrics(y_true, y_pred)

tp, fp, fn, tn = evaluator.confusion_matrix()

print("TP =", tp)
print("FP =", fp)
print("FN =", fn)
print("TN =", tn)
print("Accuracy =", evaluator.accuracy())
print("Precision =", evaluator.precision())
print("Recall =", evaluator.recall())
print("F1 =", evaluator.f1_score())

print("\nManual test:")
print("Prediction for 'this movie was amazing and wonderful':",
      model.predict("this movie was amazing and wonderful"))

print("Prediction for 'this movie was boring and terrible':",
      model.predict("this movie was boring and terrible"))