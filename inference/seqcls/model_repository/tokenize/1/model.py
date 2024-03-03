import os
import json
import numpy as np
import triton_python_backend_utils as pd_utils
from tokenizer import Tokenizer


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        config_file = os.path.join(args['model_repository'], args['model_version'], 'tokenizer_config.json')
        with open(config_file, 'r', encoding='utf-8') as in_file:
            self.tokenizer_config = json.load(in_file)
        self.tokenizer = Tokenizer.from_pretrained(self.tokenizer_config["model"])

        if self.tokenizer_config["enable_truncation"]:
            self.tokenizer.enable_truncation(self.tokenizer_config["max_length"])
        if self.tokenizer_config["enable_padding"]:
            self.tokenizer.enable_padding()

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            texts = pd_utils.get_input_tensor_by_name(request, 'text').as_numpy().tolist()
            texts = [t.decode('utf-8') for t in texts]
            outputs = self.tokenizer.encode_batch(texts)
            input_ids = np.asarray([d.attention_mask for d in outputs])
            attention_mask = np.asarray([d.ids for d in outputs])
            output_tensors = [
                pd_utils.Tensor('input_ids', input_ids),
                pd_utils.Tensor('attention_mask', attention_mask)
            ]
            if self.tokenizer_config["output_type_id"]:
                token_type_ids = np.asarray([d.type_ids for d in outputs])
                output_tensors.append(pd_utils.Tensor('token_type_ids', token_type_ids))
            response = pd_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)

        return responses
