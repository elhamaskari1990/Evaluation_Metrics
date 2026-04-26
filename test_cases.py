from evaluation_metrics import EvaluationMetrics


def print_binary_results(title, y_true, y_pred):
    print("\n" + "=" * 50)
    print(title)
    print("y_true =", y_true)
    print("y_pred =", y_pred)

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


# -------- PROVIDED TEST CASES --------

print_binary_results("Test Case 1 — Perfect Classification",
                     [1, 0, 1, 0],
                     [1, 0, 1, 0])

print_binary_results("Test Case 2 — All Wrong",
                     [1, 1, 0, 0],
                     [0, 0, 1, 1])

print_binary_results("Test Case 3 — No Predicted Positives",
                     [1, 0, 1, 0],
                     [0, 0, 0, 0])

print_binary_results("Test Case 4 — No Actual Positives",
                     [0, 0, 0, 0],
                     [1, 0, 1, 0])


# -------- SELF DESIGNED TEST CASES --------

print_binary_results("Test Case 5 — High Recall but Low Precision",
                     [1, 1, 1, 0, 0, 0],
                     [1, 1, 1, 1, 1, 0])

print_binary_results("Test Case 6 — High Precision but Low Recall",
                     [1, 1, 1, 1, 0, 0],
                     [1, 0, 0, 0, 0, 0])

print_binary_results("Test Case 7 — Rare Positives, Always Negative Prediction",
                     [0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0])


# -------- MULTI CLASS TEST --------

print("\n" + "=" * 50)
print("Multi-class Test Case")

y_true_multi = [0, 1, 2, 1, 0, 2]
y_pred_multi = [0, 2, 2, 1, 0, 1]

print("y_true =", y_true_multi)
print("y_pred =", y_pred_multi)

multi_evaluator = EvaluationMetrics(y_true_multi, y_pred_multi)
report = multi_evaluator.multiclass_report()

for cls in report:
    print("\nClass", cls)
    for key in report[cls]:
        print(key, "=", report[cls][key])