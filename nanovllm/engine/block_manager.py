"""Gestion des blocs de cache KV utilisés pendant l'inférence.

Chaque séquence occupe des "blocs" dans le cache KV. Ce gestionnaire se
charge d'allouer et de libérer ces blocs tout en essayant de réutiliser
les préfixes déjà présents en mémoire. L'objectif est de réduire au
maximum les opérations de copie et d'optimiser la bande passante mémoire.
"""

from collections import deque  # File performante pour stocker les blocs libres
import xxhash  # Librairie de hachage rapide
import numpy as np  # Manipulation efficace de tableaux

from nanovllm.engine.sequence import Sequence


class Block:
    """Représente un bloc élémentaire du cache KV."""

    def __init__(self, block_id):
        # Identifiant unique du bloc dans le pool
        self.block_id = block_id
        self.ref_count = 0  # Nombre de séquences utilisant ce bloc
        self.hash = -1      # Hash du contenu pour savoir si on peut le réutiliser
        self.token_ids = [] # Les ids de tokens actuellement stockés

    def update(self, hash: int, token_ids: list[int]):
        """Met à jour le contenu et le hash du bloc."""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """Réinitialise le bloc lorsqu'il est (ré)alloué."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Alloue et recycle les blocs de cache pour toutes les séquences."""

    def __init__(self, num_blocks: int, block_size: int):
        """Crée un pool de `num_blocks` blocs de taille fixe."""
        assert num_blocks > 0
        self.block_size = block_size  # Nombre de tokens par bloc
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # Tous les blocs
        self.hash_to_block_id: dict[int, int] = dict()  # Permet de retrouver un bloc par son hash
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # Blocs disponibles
        self.used_block_ids: set[int] = set()  # Blocs déjà alloués

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """Calcule un hash pour une liste de tokens.

        Le hash tient éventuellement compte du hash d'un bloc précédent
        (`prefix`). Cela permet de détecter rapidement si deux blocs
        consécutifs contiennent exactement le même préfixe.
        """
        h = xxhash.xxh64()  # Initialisation du calcul de hash
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))  # Prend en compte le hash précédent
        h.update(np.array(token_ids).tobytes())  # Ajoute les nouveaux tokens
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """Place un bloc dans l'ensemble des blocs utilisés."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """Libère le bloc et le remet dans la file des blocs libres."""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Vérifie s'il y a assez de blocs libres pour une séquence."""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Alloue les blocs nécessaires à la séquence et tente de
        réutiliser le cache existant.
        """
        assert not seq.block_table
        h = -1  # Hash du bloc précédent
        cache_miss = False  # Indique si l'on doit charger de nouveaux blocs
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """Libère tous les blocs associés à la séquence."""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Vérifie qu'il reste de la place pour ajouter un token."""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """Met à jour la table de blocs lors de l'ajout d'un nouveau token."""
        block_table = seq.block_table  # Liste des blocs occupés
        last_block = self.blocks[block_table[-1]]  # Dernier bloc utilisé
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
