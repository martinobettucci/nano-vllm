from dataclasses import dataclass  # Explanation: dataclass decorator simplifies storage classes
import torch  # Explanation: used for tensor types within the context


@dataclass
class Context:
    """Holds transient state required during kernel execution."""

    is_prefill: bool = False  # Explanation: flag indicating prefill vs decode stage
    cu_seqlens_q: torch.Tensor | None = None  # Explanation: cumulative sequence lengths of queries
    cu_seqlens_k: torch.Tensor | None = None  # Explanation: cumulative sequence lengths of keys
    max_seqlen_q: int = 0  # Explanation: maximum query sequence length
    max_seqlen_k: int = 0  # Explanation: maximum key sequence length
    slot_mapping: torch.Tensor | None = None  # Explanation: mapping of slots to sequence ids
    context_lens: torch.Tensor | None = None  # Explanation: per-sequence context lengths
    block_tables: torch.Tensor | None = None  # Explanation: kv-cache block table for each sequence

_CONTEXT = Context()  # Explanation: module-level storage for the current context

def get_context():
    """Pseudocode:
    1. Return the module level context instance.
    """

    return _CONTEXT  # Explanation: provide access to current context

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """Pseudocode:
    1. Build a ``Context`` instance with provided values.
    2. Replace the global context with the new instance.
    """

    global _CONTEXT  # Explanation: declare intention to modify global variable
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)  # Explanation: store new context data

def reset_context():
    """Pseudocode:
    1. Reset the global context to the default empty ``Context``.
    """

    global _CONTEXT  # Explanation: declare global variable usage
    _CONTEXT = Context()  # Explanation: create a default context instance
