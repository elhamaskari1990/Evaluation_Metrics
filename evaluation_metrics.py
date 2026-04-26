class EvaluationMetrics:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    # -------- Binary Confusion Matrix --------
    def confusion_matrix(self):
        tp = fp = fn = tn = 0

        for i in range(len(self.y_true)):
            if self.y_true[i] == 1 and self.y_pred[i] == 1:
                tp += 1
            elif self.y_true[i] == 0 and self.y_pred[i] == 1:
                fp += 1
            elif self.y_true[i] == 1 and self.y_pred[i] == 0:
                fn += 1
            else:
                tn += 1

        return tp, fp, fn, tn

    def accuracy(self):
        tp, fp, fn, tn = self.confusion_matrix()
        return (tp + tn) / len(self.y_true)

    def precision(self):
        tp, fp, fn, tn = self.confusion_matrix()
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def recall(self):
        tp, fp, fn, tn = self.confusion_matrix()
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    # -------- Multi-class Support --------
    def get_classes(self):
        classes = []

        for value in self.y_true:
            if value not in classes:
                classes.append(value)

        for value in self.y_pred:
            if value not in classes:
                classes.append(value)

        return classes

    def multiclass_confusion_matrix(self, target_class):
        tp = fp = fn = tn = 0

        for i in range(len(self.y_true)):
            if self.y_true[i] == target_class and self.y_pred[i] == target_class:
                tp += 1
            elif self.y_true[i] != target_class and self.y_pred[i] == target_class:
                fp += 1
            elif self.y_true[i] == target_class and self.y_pred[i] != target_class:
                fn += 1
            else:
                tn += 1

        return tp, fp, fn, tn

    def multiclass_report(self):
        report = {}
        classes = self.get_classes()

        for c in classes:
            tp, fp, fn, tn = self.multiclass_confusion_matrix(c)

            if tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp)

            if tp + fn == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            report[c] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }

        return report