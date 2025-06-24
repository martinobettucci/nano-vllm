from dataclasses import dataclass  # Explanation: import dataclass decorator for simple data containers


@dataclass
class SamplingParams:
    """Container for sampling-related parameters."""

    temperature: float = 1.0  # Explanation: softmax temperature used during generation
    max_tokens: int = 64  # Explanation: maximum number of tokens to generate
    ignore_eos: bool = False  # Explanation: whether to stop on the EOS token or continue generating
