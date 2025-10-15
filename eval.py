import argparse
import json
from typing import List
from sklearn.metrics import accuracy_score, roc_auc_score


def read_lines(input_file: str) -> List[str]:
    lines = []
    with open(input_file, "rb") as f:
        for line in f:
            lines.append(line.decode().strip())
    return lines


def eval_file(pred_file, label_file, score_file,metrics_output_file=None):
    pred_answers = read_lines(pred_file)
    gold_answers = read_lines(label_file)
    score_file = read_lines(score_file)
    # print(gold_answers)
    list_pred = []
    list_true = []
    for i in range(len(gold_answers)):
        if gold_answers[i] == "1":
            list_true.append(1)
            list_true.append(0)
        else:
            list_true.append(0)
            list_true.append(1)
        list_pred.append(float(score_file[i].split(',')[0].strip()))
        list_pred.append(float(score_file[i].split(',')[1].strip()))
    if len(gold_answers) != len(pred_answers):
        raise Exception("The prediction file does not contain the same number of lines as the "
                        "number of test instances.")

    accuracy = accuracy_score(gold_answers, pred_answers)
    roc_auc = roc_auc_score(list_true, list_pred)
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }
    # print(metrics)

    if metrics_output_file is not None:
        with open(metrics_output_file, "w") as f:
            f.write(json.dumps(metrics))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate ANLI predictions')
    # Required Parameters
    parser.add_argument('--labels_file', type=str, help='Location of test labels', default=None)
    parser.add_argument('--preds_file', type=str, help='Location of predictions', default=None)
    parser.add_argument('--metrics_output_file', type=str, default=None,
                        help='Location of output metrics file')

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    eval_file(args.preds_file, args.labels_file, args.metrics_output_file)
