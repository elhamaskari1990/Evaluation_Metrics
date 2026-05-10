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

            if label == "positive":
                numeric_label = 1
            else:
                numeric_label = -1

            dataset.append((text, numeric_label))

            if len(dataset) == limit:
                break

    return dataset


def build_vocab(dataset):
    vocab = {}

    for text, label in dataset:
        words = tokenize(text)

        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


class PerceptronClassifier:

    def __init__(self, epochs=10):
        self.epochs = epochs
        self.vocab = {}
        self.w = []
        self.b = 0

    def vectorize(self, text):
        vector = [0] * len(self.vocab)
        words = tokenize(text)

        for word in words:
            if word in self.vocab:
                index = self.vocab[word]
                vector[index] = 1

        return vector

    def sign(self, score):
        if score > 0:
            return 1
        else:
            return -1

    def predict_vector(self, x):
        score = self.b

        for i in range(len(x)):
            score += self.w[i] * x[i]

        return self.sign(score)

    def predict(self, text):
        x = self.vectorize(text)
        return self.predict_vector(x)

    def update(self, x, y):
        prediction = self.predict_vector(x)

        if prediction != y:
            for i in range(len(self.w)):
                self.w[i] = self.w[i] + y * x[i]

            self.b = self.b + y

    def train(self, dataset):
        self.vocab = build_vocab(dataset)
        self.w = [0] * len(self.vocab)
        self.b = 0

        for _ in range(self.epochs):
            for text, label in dataset:
                x = self.vectorize(text)
                self.update(x, label)


dataset = load_imdb_dataset("IMDB Dataset.csv", 200)

train_data = dataset[:160]
test_data = dataset[160:]

model = PerceptronClassifier(epochs=10)
model.train(train_data)

y_true = []
y_pred = []

for text, label in test_data:
    prediction = model.predict(text)

    if label == 1:
        y_true.append(1)
    else:
        y_true.append(0)

    if prediction == 1:
        y_pred.append(1)
    else:
        y_pred.append(0)

print("Predictions:")
for i in range(len(test_data)):
    print("True:", y_true[i], "| Predicted:", y_pred[i])

print("\nEvaluation:")

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