"""Interface utilisateur simplifiée pour le moteur LLM.

Cette classe hérite entièrement de :class:`~nanovllm.engine.llm_engine.LLMEngine`
afin de proposer une API identique à celle de `vLLM`. Elle ne redéfinit
aucune méthode mais permet d'importer directement ``LLM`` depuis le
paquet principal.
"""

from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """Alias pratique de :class:`LLMEngine` sans modifications."""

