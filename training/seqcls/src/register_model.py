import argparse
import os
import json
import logging

from azureml.core import Run
from transformers import AutoConfig

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Model dir')
    parser.add_argument('--onnx_dir', type=str, help='ONNX model dir', required=False)
    parser.add_argument('--registered_name', type=str, help='Registered model name', required=False)
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    config = AutoConfig.from_pretrained(args.model_dir)

    run = Run.get_context()
    properties = {
        "metrics": json.dumps(run.get_metrics()),
        "hparams": json.dumps({"num_labels": config.num_labels, "embed_size": config.hidden_size})
    }

    logger.info(f'register pytorch model {args.model_dir}')
    pytorch_model_name = f"{args.registered_name}.pytorch" if args.registered_name else "model.pytorch"
    run.upload_folder(pytorch_model_name, args.model_dir)
    pytorch_properties = {"artifact_path": pytorch_model_name}
    if args.registered_name:
        pytorch_model = run.register_model(pytorch_model_name, model_path=pytorch_model_name, properties=properties)
        pytorch_properties["model_id"] = pytorch_model.id
        pytorch_properties["model_name"] = pytorch_model.name
        pytorch_properties["model_version"] = pytorch_model.version
        pytorch_properties["model_path"] = pytorch_model_name
    properties["pytorch"] = json.dumps(pytorch_properties)

    if args.onnx_dir and os.path.exists(args.onnx_dir):
        logger.info(f'register onnx model {args.onnx_dir}')
        onnx_model_name = f"{args.registered_name}.onnx" if args.registered_name else "model.onnx"
        run.upload_folder(onnx_model_name, args.onnx_dir)
        onnx_properties = {"artifact_path": onnx_model_name}
        if args.registered_name:
            onnx_model = run.register_model(onnx_model_name, model_path=onnx_model_name, properties=properties)
            onnx_properties["model_id"] = onnx_model.id
            onnx_properties["model_name"] = onnx_model.name
            onnx_properties["model_version"] = onnx_model.version
            onnx_properties["model_path"] = onnx_model_name
        properties["onnx"] = json.dumps(onnx_properties)

    run.add_properties(properties)


if __name__ == "__main__":
    main()
