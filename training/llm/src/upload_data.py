import argparse
import logging

import azureml.core
from azureml.core import Datastore, Run

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, help='Azure datastore name')
    parser.add_argument('--blob_path', type=str, help='Azure blob path')
    parser.add_argument('--local_path', type=str, help='Local file path')
    args = parser.parse_args()

    run = Run.get_context()
    ws = run.experiment.workspace

    logger.setLevel(logging.INFO)
    logger.info(f"AzureML version: {azureml.core.VERSION}")
    logger.info(f"Workspace name: {ws.name}")
    logger.info(f"Resource group: {ws.resource_group}")
    logger.info(f"Location: {ws.location}")
    logger.info(f"Subscription id: {ws.subscription_id}")

    ds = Datastore.get(ws, datastore_name=args.ds_name)
    ds.upload_files([args.local_path], target_path=args.blob_path, overwrite=True)
    logger.info(f'upload data file from {args.local_path} to {args.ds_name}:{args.blob_path}')


if __name__ == "__main__":
    main()
