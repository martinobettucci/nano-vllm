import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        """Initialise le module Sampler."""
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """Echantillonne un jeton en fonction des logits et des températures."""
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)  # jeton le plus probable
        logits.div_(temperatures.unsqueeze(dim=1))  # applique la température
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        # bruit de Gumbel pour échantillonner selon la distribution
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
        # choix gourmand si température nulle, sinon échantillon
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
