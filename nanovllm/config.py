"""Définition et validation de la configuration globale du moteur.

La classe :class:`Config` regroupe tous les paramètres nécessaires à
l'initialisation du moteur d'inférence. Le but est de centraliser ces
options et de s'assurer qu'elles soient cohérentes dès le départ.
"""

import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """Structure contenant les paramètres du moteur.

    Attributes
    ----------
    model : str
        Chemin vers le dossier contenant les poids du modèle.
    max_num_batched_tokens : int
        Nombre maximal de tokens traités en parallèle lors du pré-remplissage.
    max_num_seqs : int
        Nombre maximal de séquences actives en même temps.
    max_model_len : int
        Longueur maximale d'une séquence acceptée par le modèle.
    gpu_memory_utilization : float
        Fraction de mémoire GPU que l'on souhaite utiliser pour le cache KV.
    tensor_parallel_size : int
        Nombre de GPU utilisés en parallèle (tensor parallelism).
    enforce_eager : bool
        Si vrai, on désactive l'exécution via CUDA Graphs pour faciliter le débogage.
    hf_config : AutoConfig | None
        Configuration HuggingFace chargée automatiquement.
    eos : int
        Identifiant du token de fin de séquence.
    kvcache_block_size : int
        Taille d'un bloc de cache KV en nombre de tokens.
    num_kvcache_blocks : int
        Nombre total de blocs disponibles pour le cache KV.
    """
    model: str
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        """Valide et complète la configuration après l'initialisation."""
        # Le chemin du modèle doit exister pour pouvoir charger les poids.
        assert os.path.isdir(self.model)
        # Chaque bloc du cache KV doit être multiple de 256 pour simplifier
        # les opérations d'alignement mémoire.
        assert self.kvcache_block_size % 256 == 0
        # Le parallélisme tensoriel est limité à 8 GPU dans notre implémentation
        # légère pour garder le code lisible.
        assert 1 <= self.tensor_parallel_size <= 8
        # On charge la configuration HuggingFace associée au modèle pour
        # récupérer automatiquement ses hyperparamètres essentiels.
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # On s'assure que la longueur de séquence demandée n'excède pas celle
        # supportée par le modèle lui-même.
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
