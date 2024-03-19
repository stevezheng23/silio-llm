import argparse
import logging
import json
import os
import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_optimized_threshold(curve, optimized_metric="precision", constraint_metric="recall", min_score=0.2):
    c_curve = [d for d in curve if d[constraint_metric] >= min_score]
    if not c_curve:
        return None
    index = np.argmax([d[optimized_metric] for d in c_curve])
    return c_curve[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, help='Evaluation file')
    parser.add_argument('--label_file', type=str, help='Label tag file')
    parser.add_argument('--search_strategy', type=str, help='Threshold search strategy', default='p/r')
    parser.add_argument('--optimized_metric', type=str, help='Optimized metric', default='precision')
    parser.add_argument('--constraint_metric', type=str, help='Constraint metric', default='recall')
    parser.add_argument('--min_score', type=float, help='Minimum score for constraint metric', default=0.2)
    parser.add_argument('--output_dir', type=str, help='Output directory')
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    eval_file = os.path.abspath(args.eval_file)
    with open(eval_file, 'r', encoding='utf-8') as in_file:
        eval_examples = []
        for line in in_file.readlines():
            d = json.loads(line.strip())
            eval_examples.append(d)

    label_file = os.path.abspath(args.label_file)
    with open(label_file, 'r', encoding='utf-8') as in_file:
        label_list = [d for d in in_file.read().splitlines() if d]
        label_dict = {k: i for i, k in enumerate(label_list)}

    threshold_optimal = {}
    if args.search_strategy == 'roc':
        logger.info('search optimal thresholds based on ROC curve')
        for label in label_list:
            label_idx = label_dict[label]
            preds = [round(d["probs"][label_idx], ndigits=4) for d in eval_examples]
            labels = [1 if d["label"] == label else 0 for d in eval_examples]
            fp_rates, tp_rates, thresholds = roc_curve(labels, preds)
            curve = [{
                "threshold": t,
                "gmean": np.sqrt(tpr * (1 - fpr)),
                "fpr": fpr,
                "tpr": tpr,
            } for fpr, tpr, t in zip(fp_rates, tp_rates, thresholds)]
            threshold_optimal[label] = {"threshold": 0.0, "gmean": 0.0, "fpr": 0.0, "tpr": 0.0}
            threshold = get_optimized_threshold(curve, args.optimized_metric, args.constraint_metric, args.min_score)
            if threshold is not None:
                threshold_optimal[label]["threshold"] = float(round(threshold["threshold"], ndigits=4))
                threshold_optimal[label]["gmean"] = float(round(threshold["gmean"], ndigits=4))
                threshold_optimal[label]["fpr"] = float(round(threshold["fpr"], ndigits=4))
                threshold_optimal[label]["tpr"] = float(round(threshold["tpr"], ndigits=4))
    else:
        logger.info('search optimal thresholds based on P/R curve')
        for label in label_list:
            label_idx = label_dict[label]
            preds = [round(d["probs"][label_idx], ndigits=4) for d in eval_examples]
            labels = [1 if d["label"] == label else 0 for d in eval_examples]
            precisions, recalls, thresholds = precision_recall_curve(labels, preds)
            curve = [{
                "threshold": t,
                "f1_score": 2 * p * r / (p + r),
                "precision": p,
                "recall": r,
            } for p, r, t in zip(precisions, recalls, thresholds) if p > 0 and r > 0]
            threshold_optimal[label] = {"threshold": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
            threshold = get_optimized_threshold(curve, args.optimized_metric, args.constraint_metric, args.min_score)
            if threshold is not None:
                threshold_optimal[label]["threshold"] = float(round(threshold["threshold"], ndigits=4))
                threshold_optimal[label]["f1_score"] = float(round(threshold["f1_score"], ndigits=4))
                threshold_optimal[label]["precision"] = float(round(threshold["precision"], ndigits=4))
                threshold_optimal[label]["recall"] = float(round(threshold["recall"], ndigits=4))

    threshold_file = os.path.join(args.output_dir, 'threshold_optimal.json')
    logger.info(f'save threshold optimal report to {threshold_file}')
    with open(threshold_file, 'w', encoding='utf-8') as out_file:
        json.dump(threshold_optimal, out_file, indent=4)


if __name__ == "__main__":
    main()
