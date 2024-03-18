import argparse
import logging
import json
import os

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, help='Evaluation file')
    parser.add_argument('--label_file', type=str, help='Label tag file')
    parser.add_argument('--threshold_file', type=str, help='Threshold optimal file')
    parser.add_argument('--default_label', type=str, help='Default label', default='default.skip')
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

    logger.info(f'save threshold optimal report to {args.threshold_file}')
    with open(args.threshold_file, 'r', encoding='utf-8') as in_file:
        threshold_optimal = json.load(in_file)

    optimal_examples = []
    for d in eval_examples:
        pred = d["pred"]
        threshold = threshold_optimal[pred]["threshold"]
        if d["score"] < threshold:
            pred = args.default_label
        example = {
            "text": d["text"],
            "pred": pred,
            "label": d["label"],
        }
        if "id" in d:
            example["id"] = d["id"]
        if "group_id" in d:
            example["group_id"] = d["group_id"]
        optimal_examples.append(example)

    optimal_eval_file = os.path.join(args.output_dir, "optimal_results.json")
    with open(optimal_eval_file, "w", encoding="utf-8") as out_file:
        for d in optimal_examples:
            out_file.write(f"{json.dumps(d)}\n")

    optimal_label_list = label_list + [args.default_label] if args.default_label not in label_list else label_list
    optimal_label_file = os.path.join(args.output_dir, "optimal_labels.txt")
    with open(optimal_label_file, "w", encoding="utf-8") as out_file:
        for l in optimal_label_list:
            out_file.write(f"{l}\n")


if __name__ == "__main__":
    main()
