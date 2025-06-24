import os  # Explanation: used to validate file system paths
from dataclasses import dataclass  # Explanation: provides dataclass decorator
from transformers import AutoConfig  # Explanation: loads Hugging Face model configuration


@dataclass
class Config:
    """LLM configuration options."""

    model: str  # Explanation: path to the local model directory
    max_num_batched_tokens: int = 32768  # Explanation: maximum number of tokens in a batch
    max_num_seqs: int = 512  # Explanation: maximum concurrent sequences
    max_model_len: int = 4096  # Explanation: requested maximum sequence length
    gpu_memory_utilization: float = 0.9  # Explanation: ratio of GPU memory that can be used
    tensor_parallel_size: int = 1  # Explanation: degree of tensor parallelism
    enforce_eager: bool = False  # Explanation: force eager mode execution
    hf_config: AutoConfig | None = None  # Explanation: Hugging Face configuration object
    eos: int = -1  # Explanation: token id used as end-of-sequence
    kvcache_block_size: int = 256  # Explanation: size of a kv-cache block
    num_kvcache_blocks: int = -1  # Explanation: number of kv-cache blocks to allocate

    def __post_init__(self):
        """Pseudocode:
        1. Ensure provided model directory exists.
        2. Validate block size and parallelism settings.
        3. Load Hugging Face model config from the directory.
        4. Clamp requested max model length using loaded config value.
        """

        assert os.path.isdir(self.model)  # Explanation: fail if the model directory does not exist
        assert self.kvcache_block_size % 256 == 0  # Explanation: block size must be multiple of 256
        assert 1 <= self.tensor_parallel_size <= 8  # Explanation: support only up to 8-way tensor parallelism
        self.hf_config = AutoConfig.from_pretrained(self.model)  # Explanation: load configuration from the model directory
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)  # Explanation: adjust max length based on config
