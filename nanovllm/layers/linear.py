import torch  # Explanation: Importation du module principal de PyTorch
from torch import nn  # Explanation: Importation de torch.nn pour les modules
import torch.nn.functional as F  # Explanation: Fonctions utilitaires de PyTorch
import torch.distributed as dist  # Explanation: Outils pour la parallélisation


def divide(numerator, denominator):  # Explanation: Divise un entier par un autre
    assert numerator % denominator == 0  # Explanation: Vérifie que la division est entière
    return numerator // denominator  # Explanation: Renvoie le quotient entier


class LinearBase(nn.Module):  # Explanation: Classe de base pour les couches linéaires

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ):
        """Initialise la couche.

        Paramètres
        ----------
        input_size : int
            Taille de la dimension d'entrée.
        output_size : int
            Taille de la dimension de sortie.
        tp_dim : int | None, optional
            Dimension utilisée pour la parallélisation tensorielle.

        Aucune forme de tenseur n'est attendue ici.
        """
        super().__init__()  # Explanation: Appel de l'initialisation de nn.Module
        self.input_size = input_size  # Explanation: Stocke la taille d'entrée
        self.output_size = output_size  # Explanation: Stocke la taille de sortie
        self.tp_dim = tp_dim  # Explanation: Dimension de parallélisation
        self.tp_rank = dist.get_rank()  # Explanation: Rang du processus courant
        self.tp_size = dist.get_world_size()  # Explanation: Nombre total de processus

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagation avant.

        Paramètres
        ----------
        x : torch.Tensor
            Tenseur d'entrée de forme ``(..., input_size)``.

        Retourne
        -------
        torch.Tensor
            Tenseur de sortie.
        """
        raise NotImplementedError  # Explanation: Doit être implémenté dans les sous-classes


class ReplicatedLinear(LinearBase):  # Explanation: Couche linéaire répliquée sur tous les processus

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """Initialise la couche.

        Paramètres
        ----------
        input_size : int
            Dimension d'entrée.
        output_size : int
            Dimension de sortie.
        bias : bool, optional
            Indique si une biais est utilisée.

        Aucun tenseur n'est requis lors de l'initialisation.
        """
        super().__init__(input_size, output_size)  # Explanation: Appel au constructeur de base
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))  # Explanation: Paramètre de poids partagé
        self.weight.weight_loader = self.weight_loader  # Explanation: Attache la fonction de chargement
        if bias:  # Explanation: Si un biais est demandé
            self.bias = nn.Parameter(torch.empty(self.output_size))  # Explanation: Crée le biais
            self.bias.weight_loader = self.weight_loader  # Explanation: Attache la fonction de chargement du biais
        else:
            self.register_parameter("bias", None)  # Explanation: Pas de biais enregistré

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data  # Explanation: Accès direct aux données du paramètre
        assert param_data.size() == loaded_weight.size()  # Explanation: Vérifie la cohérence des tailles
        param_data.copy_(loaded_weight)  # Explanation: Copie le poids chargé

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applique la transformation linéaire.

        Paramètres
        ----------
        x : torch.Tensor
            Tenseur d'entrée de forme ``(..., input_size)``.

        Retourne
        -------
        torch.Tensor
            Tenseur de sortie de forme ``(..., output_size)``.
        """
        return F.linear(x, self.weight, self.bias)  # Explanation: Produit x * W^T + b


class ColumnParallelLinear(LinearBase):  # Explanation: Couche divisant la sortie entre processus

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """Initialisation de la couche.

        Paramètres
        ----------
        input_size : int
            Dimension d'entrée.
        output_size : int
            Dimension de sortie totale.
        bias : bool, optional
            Active un biais local.
        """
        super().__init__(input_size, output_size, 0)  # Explanation: Définit la dimension parallèle colonne
        self.input_size_per_partition = input_size  # Explanation: Taille d'entrée par partition
        self.output_size_per_partition = divide(output_size, self.tp_size)  # Explanation: Découpe la sortie

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))  # Explanation: Matrice de poids par partition
        self.weight.weight_loader = self.weight_loader  # Explanation: Attache la fonction de chargement
        if bias:  # Explanation: Si un biais local est demandé
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))  # Explanation: Crée le biais local
            self.bias.weight_loader = self.weight_loader  # Explanation: Attache la fonction de chargement du biais
        else:
            self.register_parameter("bias", None)  # Explanation: Aucun biais n'est enregistré

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data  # Explanation: Accès au stockage du paramètre
        shard_size = param_data.size(self.tp_dim)  # Explanation: Taille de la portion locale
        start_idx = self.tp_rank * shard_size  # Explanation: Début de la portion dans le poids total
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)  # Explanation: Coupe le poids chargé
        assert param_data.size() == loaded_weight.size()  # Explanation: Vérifie la taille
        param_data.copy_(loaded_weight)  # Explanation: Copie la portion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applique la partie locale de la projection.

        Paramètres
        ----------
        x : torch.Tensor
            Tenseur d'entrée ``(..., input_size)``.

        Retourne
        -------
        torch.Tensor
            Résultat local ``(..., output_size / tp_size)``.
        """
        return F.linear(x, self.weight, self.bias)  # Explanation: Produit la transformation locale


class MergedColumnParallelLinear(ColumnParallelLinear):  # Explanation: Combine plusieurs projections en une

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        """Initialisation.

        Paramètres
        ----------
        input_size : int
            Taille de l'entrée.
        output_sizes : list[int]
            Tailles de sortie de chaque projection fusionnée.
        bias : bool, optional
            Active un biais sur chaque sortie.
        """
        self.output_sizes = output_sizes  # Explanation: Stocke les tailles individuelles
        super().__init__(input_size, sum(output_sizes), bias=bias)  # Explanation: Initialise la classe parente avec la somme des sorties

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data  # Explanation: Accès direct aux données du paramètre
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size  # Explanation: Début du segment à charger
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size  # Explanation: Taille du segment à charger
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)  # Explanation: Sélectionne la portion du paramètre
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]  # Explanation: Sépare le poids en morceaux
        assert param_data.size() == loaded_weight.size()  # Explanation: Vérifie la correspondance des tailles
        param_data.copy_(loaded_weight)  # Explanation: Copie la portion dans le paramètre


class QKVParallelLinear(ColumnParallelLinear):  # Explanation: Projette Q, K et V en parallèle

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        """Initialisation de la projection QKV.

        Paramètres
        ----------
        hidden_size : int
            Dimension cachée en entrée.
        head_size : int
            Taille d'une tête d'attention.
        total_num_heads : int
            Nombre total de têtes Q.
        total_num_kv_heads : int | None, optional
            Nombre total de têtes K/V.
        bias : bool, optional
            Active les biais.
        """
        self.head_size = head_size  # Explanation: Taille d'une tête
        self.total_num_heads = total_num_heads  # Explanation: Nombre de têtes Q total
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads  # Explanation: Nombre de têtes K/V total
        tp_size = dist.get_world_size()  # Explanation: Récupère le nombre de processus
        self.num_heads = divide(self.total_num_heads, tp_size)  # Explanation: Têtes Q par processus
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)  # Explanation: Têtes K/V par processus
        input_size = hidden_size  # Explanation: Dimension d'entrée pour la projection
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size  # Explanation: Taille de sortie totale
        super().__init__(input_size, output_size, bias)  # Explanation: Initialise la classe parente

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data  # Explanation: Accès direct aux données
        assert loaded_shard_id in ["q", "k", "v"]  # Explanation: Type de shard attendu
        if loaded_shard_id == "q":  # Explanation: Poids de la requête
            shard_size = self.num_heads * self.head_size  # Explanation: Taille de la portion Q
            shard_offset = 0  # Explanation: Début du segment Q
        elif loaded_shard_id == "k":  # Explanation: Poids de la clé
            shard_size = self.num_kv_heads * self.head_size  # Explanation: Taille de la portion K
            shard_offset = self.num_heads * self.head_size  # Explanation: Décalage après Q
        else:  # Explanation: Poids de la valeur
            shard_size = self.num_kv_heads * self.head_size  # Explanation: Taille de la portion V
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size  # Explanation: Décalage après Q et K
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)  # Explanation: Sélectionne la partie du paramètre
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]  # Explanation: Coupe le poids et sélectionne la part locale
        assert param_data.size() == loaded_weight.size()  # Explanation: Vérifie la taille
        param_data.copy_(loaded_weight)  # Explanation: Copie la portion


class RowParallelLinear(LinearBase):  # Explanation: Répartit l'entrée entre les processus

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        """Initialisation de la couche.

        Paramètres
        ----------
        input_size : int
            Dimension d'entrée totale.
        output_size : int
            Dimension de sortie.
        bias : bool, optional
            Active un biais.
        """
        super().__init__(input_size, output_size, 1)  # Explanation: Dimension parallèle sur les lignes
        self.input_size_per_partition = divide(input_size, self.tp_size)  # Explanation: Taille d'entrée par processus
        self.output_size_per_partition = output_size  # Explanation: Chaque processus produit la sortie complète

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))  # Explanation: Poids local
        self.weight.weight_loader = self.weight_loader  # Explanation: Fonction de chargement
        if bias:  # Explanation: Ajout d'un biais si nécessaire
            self.bias = nn.Parameter(torch.empty(self.output_size))  # Explanation: Biais partagé
            self.bias.weight_loader = self.weight_loader  # Explanation: Attache la fonction de chargement du biais
        else:
            self.register_parameter("bias", None)  # Explanation: Pas de biais

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data  # Explanation: Stockage du paramètre
        shard_size = param_data.size(self.tp_dim)  # Explanation: Taille locale
        start_idx = self.tp_rank * shard_size  # Explanation: Position de début
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)  # Explanation: Extrait la portion correspondante
        assert param_data.size() == loaded_weight.size()  # Explanation: Vérifie la taille
        param_data.copy_(loaded_weight)  # Explanation: Copie la portion dans le paramètre

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applique la partie de la projection en parallèle de lignes.

        Paramètres
        ----------
        x : torch.Tensor
            Tenseur d'entrée ``(..., input_size / tp_size)``.

        Retourne
        -------
        torch.Tensor
            Tenseur de sortie ``(..., output_size)``.
        """
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)  # Explanation: Multiplie la partie locale et ajoute le biais uniquement sur le rang 0
        if self.tp_size > 1:  # Explanation: Si plusieurs processus sont utilisés
            dist.all_reduce(y)  # Explanation: Agrège les résultats
        return y  # Explanation: Renvoie le résultat final
