import argparse
import os
import json
import time
import logging

import mlflow
import mlflow.azureml

import azureml.core
from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import AzureCliAuthentication

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


supported_entry_point = {
    "main",
    "llm_train",
    "llm_infer",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry_point', type=str, help='Entry point', default='main')
    parser.add_argument('--base_file', type=str, help='Base experiment file', default='./llm_experiment.json')
    parser.add_argument('--input_file', type=str, help='Input experiment file', default='./input_experiment.json')
    parser.add_argument('--output_file', type=str, help='Output experiment file', default='./output_experiment.json')
    args = parser.parse_args()

    logger.info(f"MLflow version: {mlflow.version.VERSION}")
    logger.info(f"AzureML version: {azureml.core.VERSION}")

    if args.entry_point not in supported_entry_point:
        raise ValueError("entry point not supported")
    if not os.path.exists(args.base_file):
        raise FileNotFoundError("base experiment file not found")

    logger.info(f"Read base experiment config from: {args.base_file}")
    script_params = {}
    submit_params = {}
    with open(args.base_file, "r", encoding="utf-8") as in_file:
        experiments = json.load(in_file)
        if not experiments:
            raise FileNotFoundError("base experiment config not found")
        script_params.update(experiments[-1]["hparam"])
        script_params.update(experiments[-1]["data"])
        submit_params.update(experiments[-1]["platform"])

    if os.path.exists(args.input_file):
        logger.info(f"Read input experiment config from: {args.input_file}")
        with open(args.input_file, "r", encoding="utf-8") as in_file:
            experiments = json.load(in_file)
            script_params["train_run_id"] = experiments["run_id"]
            if "pytorch" in experiments:
                script_params["artifact_path"] = experiments["pytorch"]["artifact_path"]

    ws = Workspace.get(name=submit_params["aml_workspace_name"],
                       subscription_id=submit_params["aml_subscription_id"],
                       resource_group=submit_params["aml_resource_group"],
                       auth=AzureCliAuthentication())

    logger.info(f"Workspace name: {ws.name}")
    logger.info(f"Resource group: {ws.resource_group}")
    logger.info(f"Location: {ws.location}")
    logger.info(f"Subscription id: {ws.subscription_id}")

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    logger.info(f"Current tracking URI: {mlflow.get_tracking_uri()}")

    mlflow.set_experiment(submit_params["aml_experiment_name"])

    backend_config = {
        "COMPUTE": submit_params["aml_compute_target"],
        "USE_CONDA": True
    }

    mlflow_run = mlflow.projects.run(uri=".",
                                     entry_point=args.entry_point,
                                     parameters=script_params,
                                     backend="azureml",
                                     backend_config=backend_config,
                                     synchronous=False)

    logger.info(f"MLFlow run ID: {mlflow_run.run_id}")
    run_status = mlflow_run.get_status().lower()
    while run_status in ["running", "scheduled", "preparing", "queued"]:
        logger.info(f"Current job status: {run_status}")
        time.sleep(60)
        run_status = mlflow_run.get_status().lower()
    logger.info(f"Final job status: {run_status}")
    mlflow_run.wait()

    aml_run = Run(Experiment(ws, submit_params["aml_experiment_name"]), mlflow_run.run_id)
    aml_run_properties = aml_run.get_properties()
    experiment_output = {"run_id": mlflow_run.run_id}
    if "hparams" in aml_run_properties:
        experiment_output["hparams"] = json.loads(aml_run_properties["hparams"])
    if "pytorch" in aml_run_properties:
        experiment_output["pytorch"] = json.loads(aml_run_properties["pytorch"])
    if "onnx" in aml_run_properties:
        experiment_output["onnx"] = json.loads(aml_run_properties["onnx"])

    logger.info(f"Write output experiment config to: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as out_file:
        json.dump(experiment_output, out_file, indent=2)
    logger.info(f"Generate experiment JSON file: {json.dumps(experiment_output, indent=2)}")


if __name__ == "__main__":
    main()
