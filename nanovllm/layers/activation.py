"""Fonctions d'activation utilisées dans le modèle."""

import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """Implémente l'opération SiLU suivie d'une multiplication."""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applique ``SiLU(x1) * x2`` après avoir séparé le tensor en deux."""
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
