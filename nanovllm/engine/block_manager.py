from collections import deque  # Explanation: import a double-ended queue for block IDs
import xxhash  # Explanation: hash library used for block caching
import numpy as np  # Explanation: numerical library for converting token lists

from nanovllm.engine.sequence import Sequence  # Explanation: sequence object storing token info


class Block:
    """Classe gérant un bloc de cache.

    Arguments pseudocode:
        block_id: identifiant entier du bloc.
    Fonctionnement:
        - stocke l'état et le contenu du bloc.
    """

    def __init__(self, block_id):
        self.block_id = block_id  # Explanation: identifiant du bloc
        self.ref_count = 0  # Explanation: nombre de références actuelles
        self.hash = -1  # Explanation: valeur de hachage du bloc
        self.token_ids = []  # Explanation: liste d'identifiants de tokens

    def update(self, hash: int, token_ids: list[int]):
        """Met à jour le bloc avec un nouveau hash et de nouveaux tokens."""
        self.hash = hash  # Explanation: sauvegarde le nouveau hachage
        self.token_ids = token_ids  # Explanation: enregistre les tokens chargés

    def reset(self):
        """Réinitialise le bloc pour une nouvelle utilisation."""
        self.ref_count = 1  # Explanation: bloc occupé par un appelant
        self.hash = -1  # Explanation: efface l'ancien hachage
        self.token_ids = []  # Explanation: vide la liste des tokens


class BlockManager:
    """Gère l'allocation des blocs de cache.

    Pseudocode des paramètres:
        num_blocks: nombre total de blocs disponibles.
        block_size: taille d'un bloc en tokens.
    Logique:
        - suit les blocs libres et utilisés pour le cache.
    """

    def __init__(self, num_blocks: int, block_size: int):
        """Initialise la structure de gestion des blocs."""
        assert num_blocks > 0  # Explanation: garantit qu'il y a au moins un bloc
        self.block_size = block_size  # Explanation: nombre de tokens par bloc
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # Explanation: liste des blocs alloués
        self.hash_to_block_id: dict[int, int] = dict()  # Explanation: map hash -> bloc
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # Explanation: pile des blocs libres
        self.used_block_ids: set[int] = set()  # Explanation: ensemble des blocs utilisés

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """Calcule un hash des tokens pour dédupliquer."""
        h = xxhash.xxh64()  # Explanation: initialisation de l'algorithme de hash
        if prefix != -1:  # Explanation: ajoute le hash du bloc précédent si présent
            h.update(prefix.to_bytes(8, "little"))  # Explanation: incorpore le préfixe
        h.update(np.array(token_ids).tobytes())  # Explanation: ajoute les tokens au calcul
        return h.intdigest()  # Explanation: renvoie la valeur entière du hash

    def _allocate_block(self, block_id: int) -> Block:
        """Alloue un bloc libre donné."""
        block = self.blocks[block_id]  # Explanation: récupère l'objet Block
        assert block.ref_count == 0  # Explanation: vérifie que le bloc est libre
        block.reset()  # Explanation: prépare le bloc à être utilisé
        self.free_block_ids.remove(block_id)  # Explanation: retire des blocs libres
        self.used_block_ids.add(block_id)  # Explanation: marque le bloc comme utilisé
        return self.blocks[block_id]  # Explanation: renvoie le bloc alloué

    def _deallocate_block(self, block_id: int) -> Block:
        """Libère un bloc utilisé."""
        assert self.blocks[block_id].ref_count == 0  # Explanation: s'assure que personne n'utilise le bloc
        self.used_block_ids.remove(block_id)  # Explanation: supprime des blocs utilisés
        self.free_block_ids.append(block_id)  # Explanation: remet dans la file des blocs libres

    def can_allocate(self, seq: Sequence) -> bool:
        """Vérifie si la séquence peut obtenir assez de blocs."""
        return len(self.free_block_ids) >= seq.num_blocks  # Explanation: compare blocs libres et besoin

    def allocate(self, seq: Sequence):
        """Attribue des blocs à une séquence."""
        assert not seq.block_table  # Explanation: la séquence ne doit pas déjà avoir de blocs
        h = -1  # Explanation: hash cumulatif des blocs
        cache_miss = False  # Explanation: indique si l'on doit allouer un nouveau bloc
        for i in range(seq.num_blocks):  # Explanation: itère sur chaque bloc nécessaire
            token_ids = seq.block(i)  # Explanation: récupère le contenu du bloc
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1  # Explanation: calcule le hash si bloc complet
            block_id = self.hash_to_block_id.get(h, -1)  # Explanation: tente de réutiliser un bloc existant
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:  # Explanation: vérifie la validité du cache
                cache_miss = True  # Explanation: on doit créer un nouveau bloc
            if cache_miss:
                block_id = self.free_block_ids[0]  # Explanation: choisit un bloc libre
                block = self._allocate_block(block_id)  # Explanation: alloue ce bloc
            else:
                seq.num_cached_tokens += self.block_size  # Explanation: utilise le bloc mis en cache
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]  # Explanation: bloc déjà utilisé
                    block.ref_count += 1  # Explanation: incrémente la référence
                else:
                    block = self._allocate_block(block_id)  # Explanation: alloue le bloc absent
            if h != -1:
                block.update(h, token_ids)  # Explanation: met à jour le bloc avec le hash
                self.hash_to_block_id[h] = block_id  # Explanation: enregistre la correspondance hash -> bloc
            seq.block_table.append(block_id)  # Explanation: ajoute ce bloc au tableau de la séquence

    def deallocate(self, seq: Sequence):
        """Libère tous les blocs associés à une séquence."""
        for block_id in reversed(seq.block_table):  # Explanation: traite les blocs du dernier au premier
            block = self.blocks[block_id]  # Explanation: récupère l'objet block
            block.ref_count -= 1  # Explanation: décrémente le compteur de références
            if block.ref_count == 0:  # Explanation: si plus utilisé
                self._deallocate_block(block_id)  # Explanation: remet le bloc dans la pile libre
        seq.num_cached_tokens = 0  # Explanation: réinitialise le nombre de tokens en cache
        seq.block_table.clear()  # Explanation: vide la table des blocs

    def can_append(self, seq: Sequence) -> bool:
        """Teste si un bloc supplémentaire peut être ajouté."""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)  # Explanation: besoin d'un bloc libre si séquence commence un nouveau bloc

    def may_append(self, seq: Sequence):
        """Ajoute éventuellement un bloc en fin de séquence."""
        block_table = seq.block_table  # Explanation: liste des blocs de la séquence
        last_block = self.blocks[block_table[-1]]  # Explanation: obtient le dernier bloc
        if len(seq) % self.block_size == 1:  # Explanation: début d'un nouveau bloc
            assert last_block.hash != -1  # Explanation: le bloc précédent doit être complet
            block_id = self.free_block_ids[0]  # Explanation: prend un bloc libre
            self._allocate_block(block_id)  # Explanation: alloue ce bloc
            block_table.append(block_id)  # Explanation: ajoute à la séquence
        elif len(seq) % self.block_size == 0:  # Explanation: bloc juste rempli
            assert last_block.hash == -1  # Explanation: le bloc n'a pas encore de hash
            token_ids = seq.block(seq.num_blocks-1)  # Explanation: récupère les tokens du bloc plein
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1  # Explanation: hash du bloc précédent s'il existe
            h = self.compute_hash(token_ids, prefix)  # Explanation: calcule le hash du bloc
            last_block.update(h, token_ids)  # Explanation: enregistre hash et contenu
            self.hash_to_block_id[h] = last_block.block_id  # Explanation: mappe le hash au bloc
        else:
            assert last_block.hash == -1  # Explanation: au milieu d'un bloc, rien à faire
