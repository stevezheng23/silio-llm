import os
import json
import numpy as np
import onnxruntime_genai as og
import triton_python_backend_utils as pd_utils


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

        model_path = os.path.join(args['model_repository'], args['model_version'], 'model')
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)

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
            messages = pd_utils.get_input_tensor_by_name(request, 'messages').as_numpy().tolist()
            prompt = self._format_prompt(messages)
            input_tokens = self.tokenizer.encode(prompt)
            arguments = pd_utils.get_input_tensor_by_name(request, 'arguments').as_numpy().tolist()
            output_results = []
            for arg in arguments:
                argument = json.loads(arg.decode('utf-8'))
                params = og.GeneratorParams(self.model)
                params.set_search_options(
                    max_length=argument['max_length'],
                    top_p=argument['top_p'],
                    top_k=argument['top_k'],
                    temperature=argument['temperature'],
                    repetition_penalty=argument['repetition_penalty'],
                )
                params.input_ids = input_tokens
                output_tokens = self.model.generate(params)
                result = {
                    'role': 'assistant',
                    'content': self.tokenizer.decode(output_tokens)
                }
                output_results.append(json.dumps(result))
            output_tensors = [
                pd_utils.Tensor('results', np.array(output_results, dtype=object)),
            ]
            response = pd_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)

        return responses

    def _format_messages(self, messages):
        texts = []
        for msg in messages:
            message = json.loads(msg.decode('utf-8'))
            role, content = message['role'], message['content']
            texts.append(f'<|{role}|>\n{content}<|end|>')
        texts.append('<|assistant|>')
        prompt = '\n'.join(texts)
        return prompt
