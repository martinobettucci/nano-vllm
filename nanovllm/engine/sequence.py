from copy import copy  # Explanation: pour dupliquer la liste de tokens sans la modifier
from enum import Enum, auto  # Explanation: fournit une énumération pour suivre l'état de la séquence
from itertools import count  # Explanation: génère un compteur infini pour identifier les séquences

from nanovllm.sampling_params import SamplingParams  # Explanation: contient les paramètres d'échantillonnage


class SequenceStatus(Enum):  # Explanation: énumère les différents états d'une séquence
    WAITING = auto()  # Explanation: indique que la séquence attend son tour
    RUNNING = auto()  # Explanation: indique que la séquence est en cours d'exécution
    FINISHED = auto()  # Explanation: indique que la séquence est terminée


class Sequence:  # Explanation: représente une requête avec son état et ses tokens
    block_size = 256  # Explanation: nombre de tokens par bloc de cache
    counter = count()  # Explanation: compteur global pour attribuer un identifiant unique

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        """Initialise la séquence avec les tokens fournis et les paramètres d'échantillonnage."""
        self.seq_id = next(Sequence.counter)  # Explanation: identifiant unique de la séquence
        self.status = SequenceStatus.WAITING  # Explanation: état initial de la séquence
        self.token_ids = copy(token_ids)  # Explanation: copie la liste de tokens pour éviter les modifications externes
        self.last_token = token_ids[-1]  # Explanation: dernier token pour les étapes de décodage
        self.num_tokens = len(self.token_ids)  # Explanation: compteur du nombre total de tokens
        self.num_prompt_tokens = len(token_ids)  # Explanation: nombre de tokens appartenant au prompt
        self.num_cached_tokens = 0  # Explanation: nombre de tokens déjà présents dans le cache
        self.block_table = []  # Explanation: liste des blocs alloués dans le cache
        self.temperature = sampling_params.temperature  # Explanation: température d'échantillonnage
        self.max_tokens = sampling_params.max_tokens  # Explanation: nombre maximum de tokens générés
        self.ignore_eos = sampling_params.ignore_eos  # Explanation: ignorer ou non le symbole de fin de séquence

    def __len__(self):
        """Retourne le nombre total de tokens de la séquence."""
        return self.num_tokens  # Explanation: longueur actuelle de la séquence

    def __getitem__(self, key):
        """Accède à un ou plusieurs tokens via l'index fourni."""
        return self.token_ids[key]  # Explanation: récupération d'un élément par index

    @property
    def is_finished(self):
        """Indique si la séquence est terminée."""
        return self.status == SequenceStatus.FINISHED  # Explanation: comparaison de l'état avec FINISHED

    @property
    def num_completion_tokens(self):
        """Nombre de tokens générés en plus du prompt."""
        return self.num_tokens - self.num_prompt_tokens  # Explanation: différence entre tokens totaux et tokens du prompt

    @property
    def prompt_token_ids(self):
        """Renvoie les tokens du prompt d'origine."""
        return self.token_ids[:self.num_prompt_tokens]  # Explanation: tranche correspondant au prompt

    @property
    def completion_token_ids(self):
        """Renvoie les tokens générés après le prompt."""
        return self.token_ids[self.num_prompt_tokens:]  # Explanation: tranche correspondant à la complétion

    @property
    def num_cached_blocks(self):
        """Nombre de blocs déjà présents dans le cache."""
        return self.num_cached_tokens // self.block_size  # Explanation: calcul en fonction de la taille d'un bloc

    @property
    def num_blocks(self):
        """Calcule le nombre de blocs nécessaires pour la séquence."""
        return (self.num_tokens + self.block_size - 1) // self.block_size  # Explanation: arrondi supérieur sur la taille de bloc

    @property
    def last_block_num_tokens(self):
        """Donne le nombre de tokens dans le dernier bloc."""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size  # Explanation: reste de la division en blocs

    def block(self, i):
        """Retourne les tokens contenus dans le bloc indexé."""
        assert 0 <= i < self.num_blocks  # Explanation: vérifie que l'index est valide
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]  # Explanation: découpe la liste en fonction de la taille du bloc

    def append_token(self, token_id: int):
        """Ajoute un token à la fin de la séquence."""
        self.token_ids.append(token_id)  # Explanation: ajoute le nouveau token à la liste
        self.last_token = token_id  # Explanation: met à jour le dernier token connu
        self.num_tokens += 1  # Explanation: incrémente le compteur total de tokens

    def __getstate__(self):
        """Prépare l'état sérialisable de la séquence."""
        state = {
            "num_tokens": self.num_tokens,  # Explanation: nombre actuel de tokens
            "num_prompt_tokens": self.num_prompt_tokens,  # Explanation: longueur du prompt
            "num_cached_tokens": self.num_cached_tokens,  # Explanation: tokens déjà placés en cache
            "block_table": self.block_table,  # Explanation: correspondances entre blocs et cache
        }
        if self.num_completion_tokens == 0:  # Explanation: cas où la complétion n'a pas encore commencé
            state["token_ids"] = self.token_ids  # Explanation: on sauvegarde tous les tokens
        else:
            state["last_token"] = self.last_token  # Explanation: seule la dernière génération est nécessaire
        return state  # Explanation: dictionnaire utilisé pour la sérialisation
