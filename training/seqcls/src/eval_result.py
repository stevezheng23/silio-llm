import argparse
import logging
import json
import os
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, help='Evaluation file')
    parser.add_argument('--label_file', type=str, help='Label tag file')
    parser.add_argument('--output_dir', type=str, help='Output dir')
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
        label_ids = list(range(len(label_list)))

    logger.info('generate classification report')
    preds = [label_dict[d["pred"]] for d in eval_examples]
    labels = [label_dict[d["label"]] for d in eval_examples]
    report = classification_report(y_true=labels,
                                   y_pred=preds,
                                   labels=label_ids,
                                   target_names=label_list,
                                   digits=4)

    report_file = os.path.join(args.output_dir, 'classification_report.txt')
    logger.info(f'save classification report to {report_file}')
    with open(report_file, 'w', encoding='utf-8') as out_file:
        out_file.write(report)

    logger.info('generate confusion matrix')
    preds = [d["pred"] for d in eval_examples]
    labels = [d["label"] for d in eval_examples]
    matrix = confusion_matrix(y_true=labels,
                              y_pred=preds,
                              labels=label_list)

    _, ax = plt.subplots(figsize=(15, 15))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                  display_labels=label_list)
    disp.plot(xticks_rotation="vertical", ax=ax)

    matrix_file = os.path.join(args.output_dir, 'confusion_matrix.png')
    logger.info(f'save confusion matrix figure to {matrix_file}')
    plt.savefig(matrix_file)


if __name__ == "__main__":
    main()
