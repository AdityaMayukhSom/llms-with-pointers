1. First extract the required tensors into a seperate list, then stack them together. Do not create the stacked tensor and slice it. This will reduce GPU memory bottleneck. `llama.py`             
    ```py
    layer_hidden_states = torch.stack(outputs.hidden_states, dim=1).clone()
    llama_hidden_state = layer_hidden_states[:, -1:, -1, :]
    dola_hidden_states = layer_hidden_states[:, self._llama_utils._dola_candidate_indices, -1, :]
    ```

