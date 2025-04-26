import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

class LLMExtractor:
    """
    Class for loading LLM weights/activations (via HuggingFace) and converting them into a point cloud for QTI Memory.
    Supports loading large models in parts (batching).
    """
    def __init__(self, model_name_or_path, device='cpu', dtype=torch.float32):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Loads the model and tokenizer from HuggingFace.
        """
        self.model = AutoModel.from_pretrained(self.model_name_or_path, torch_dtype=self.dtype, device_map='auto' if self.device=='auto' else None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def extract_weights(self, layer_filter=None, max_elements=None):
        """
        Extracts model weights (or only selected layers).
        :param layer_filter: callable, filter by layer name (e.g., lambda n: 'attention' in n)
        :param max_elements: int, maximum number of elements for the point cloud (for RAM limits)
        :return: np.ndarray, shape=(n_points, weight_dim)
        """
        if self.model is None:
            self.load_model()
        weights = []
        for name, param in self.model.named_parameters():
            if layer_filter is not None and not layer_filter(name):
                continue
            arr = param.detach().cpu().numpy().flatten()
            if max_elements is not None and len(arr) > max_elements:
                arr = arr[:max_elements]
            weights.append(arr)
        # Convert to point cloud (each layer is a point in weight space)
        # For large models: batch/stream
        min_len = min(len(w) for w in weights)
        points = np.stack([w[:min_len] for w in weights], axis=0)
        return points

    def stream_to_memory(self, memory, layer_filter=None, max_elements=None, batch_size=100):
        """
        Streams model weights into QTI memory (Memory.add_point).
        :param memory: Memory object
        :param layer_filter: filter by layer name
        :param max_elements: maximum number of elements per point
        :param batch_size: number of points to add at once
        """
        if self.model is None:
            self.load_model()
        batch = []
        for name, param in self.model.named_parameters():
            if layer_filter is not None and not layer_filter(name):
                continue
            arr = param.detach().cpu().numpy().flatten()
            if max_elements is not None and len(arr) > max_elements:
                arr = arr[:max_elements]
            batch.append(arr)
            if len(batch) >= batch_size:
                min_len = min(len(w) for w in batch)
                for w in batch:
                    memory.add_point(w[:min_len])
                batch = []
        # Add remaining batch
        if batch:
            min_len = min(len(w) for w in batch)
            for w in batch:
                memory.add_point(w[:min_len]) 