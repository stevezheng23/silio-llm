import argparse
import json
import os
import logging

import numpy as np
import tritonclient.grpc

from pathlib import Path

logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def test_generate(examples, model_endpoint, model_name, model_version, timeout):
    triton_client = tritonclient.grpc.InferenceServerClient(url=model_endpoint,
                                                            network_timeout=timeout,
                                                            connection_timeout=timeout,
                                                            verbose=False)
    model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version, as_json=True)
    logger.info(model_metadata)

    results = []
    for example in examples:
        messages = [json.dumps(d) for d in example["input"]["messages"]]
        input_messages = tritonclient.grpc.InferInput(name="messages", shape=(len(messages),), datatype="BYTES")
        input_messages.set_data_from_numpy(np.array(messages, dtype=object))

        arguments = [json.dumps(d) for d in example["input"]["arguments"]]
        input_arguments = tritonclient.grpc.InferInput(name="arguments", shape=(len(arguments),), datatype="BYTES")
        input_arguments.set_data_from_numpy(np.array(arguments, dtype=object))

        output_results = tritonclient.grpc.InferRequestedOutput(name="results", binary_data=True)
        infer_results = triton_client.infer(model_name=model_name,
                                            model_version=model_version,
                                            inputs=[input_messages, input_arguments],
                                            outputs=[output_results])
        result = [json.loads(d.decode("utf-8")) for d in infer_results.as_numpy("results").tolist()]
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, help='task name')
    parser.add_argument('--test_file', type=str, help='test file')
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--model_endpoint', type=str, help='model endpoint')
    parser.add_argument('--model_name', type=str, help='modle name')
    parser.add_argument('--model_version', type=str, help='model version')
    parser.add_argument('--timeout', type=float, help='timeout')
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    with open(args.test_file, "r", encoding="utf-8") as in_file:
        test_examples = json.load(in_file)

    if args.task_name == "generate":
        results = test_generate(test_examples,
                                args.model_endpoint,
                                args.model_name,
                                args.model_version,
                                args.timeout)
    else:
        raise ValueError(f"unsupported task: {args.task_name}")

    logger.info(f'save results to {args.output_dir}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(args.output_dir, "test_results.json")
    with open(output_file, "w", encoding="utf-8") as out_file:
        for d in results:
            line = json.dumps(d)
            out_file.write(f"{line}\n")


if __name__ == "__main__":
    main()
