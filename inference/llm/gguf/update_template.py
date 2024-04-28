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
    parser.add_argument("--n_ctx", type=int, help="context length", required=False)
    parser.add_argument("--n_threads", type=str, help="number of threads", required=False)
    parser.add_argument("--n_gpu_layers", type=int, help="number of gpu layers", required=False)
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    logger.info(f"load template file from: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as in_file:
        tmpl = in_file.read()

    replaces = {}
    if args.n_ctx:
        replaces["n_ctx"] = args.n_ctx
    if args.n_threads:
        replaces["n_threads"] = args.n_threads
    if args.n_gpu_layers:
        replaces["n_gpu_layers"] = args.n_gpu_layers

    t = Template(tmpl)
    res = t.substitute(replaces)

    logger.info(f"save updated file to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as out_file:
        out_file.write(res)


if __name__ == "__main__":
    main()