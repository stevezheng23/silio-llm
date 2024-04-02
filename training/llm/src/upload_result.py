import argparse
import logging

from azureml.core import Run

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, help='Result directory')
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    run = Run.get_context()
    logger.info(f'upload results from {args.result_dir}')
    run.upload_folder("result", args.result_dir)


if __name__ == "__main__":
    main()
