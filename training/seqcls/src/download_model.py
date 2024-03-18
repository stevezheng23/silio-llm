import argparse
import os
import logging

from azureml.core import Run

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, help='Train run ID')
    parser.add_argument('--artifact_path', type=str, help='Artifact path')
    parser.add_argument('--local_dir', type=str, help='Local directory')
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    run = Run.get_context()
    train_run = Run(run.experiment, args.run_id)
    if not os.path.exists(args.local_dir):
        os.mkdir(args.local_dir)
    train_run.download_files(args.artifact_path, args.local_dir)
    logger.info(f'download model artifacts to {args.local_dir}')
    logger.info(f'list all artifacts: {os.listdir(args.local_dir)}')


if __name__ == "__main__":
    main()
