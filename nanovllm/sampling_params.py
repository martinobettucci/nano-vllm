"""Paramètres de génération pour l'échantillonnage.

Cette structure simple est transmise à chaque requête afin de contrôler
le comportement du moteur lors de la production de nouveaux tokens.
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Options de sampling pour une séquence."""

    temperature: float = 1.0  # Température de l'échantillonnage (0 = greedy)
    max_tokens: int = 64  # Nombre maximum de tokens générés
    ignore_eos: bool = False  # Ne pas arrêter sur le token EOS
