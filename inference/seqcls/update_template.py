import argparse
import logging
from string import Template

logging.basicConfig(format="%(levelname)s: %(asctime)s %(message)s",
                    datefmt="%m/%d/%Y %I:%M:$S %p",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input file")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--num_labels", type=int, help="number of labels", required=False)
    parser.add_argument("--model", type=str, help="model", required=False)
    parser.add_argument("--max_length", type=int, help="max length", required=False)
    parser.add_argument("--output_type_ids", type=bool, help="whether to output type ids or not", required=False)
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    logger.info(f"load template file from: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as in_file:
        tmpl = in_file.read()

    replaces = {}
    if args.num_labes:
        replaces["num_labels"] = args.num_labels
    if args.model:
        replaces["model"] = args.model
    if args.max_length:
        replaces["max_length"] = args.max_length
    if args.output_type_ids:
        replaces["output_type_ids"] = str(args.output_type_ids).lower()

    t = Template(tmpl)
    res = t.substitute(replaces)

    logger.info(f"save updated file to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as out_file:
        out_file.write(res)


if __name__ == "__main__":
    main()