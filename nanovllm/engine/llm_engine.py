"""Implémentation principale du moteur d'inférence.

Cette classe orchestre l'exécution d'un modèle en gérant la création
des différents processus (un par GPU), la planification des séquences et
les échanges avec le *ModelRunner*.
"""

import atexit  # Pour exécuter un nettoyage à la sortie du programme
from dataclasses import fields  # Utilisé pour inspecter les champs du dataclass Config
from time import perf_counter  # Mesure précise du temps
from tqdm.auto import tqdm  # Barre de progression pratique
from transformers import AutoTokenizer  # Tokeniseur HuggingFace
import torch.multiprocessing as mp  # Gestion des processus GPU

from nanovllm.config import Config  # Karen: structure de configuration centrale
from nanovllm.sampling_params import SamplingParams  # Karen: paramètres de génération
from nanovllm.engine.sequence import Sequence  # Karen: représentation d'une requête
from nanovllm.engine.scheduler import Scheduler  # Karen: planifie l'ordre de traitement
from nanovllm.engine.model_runner import ModelRunner  # Karen: exécute le modèle sur GPU


class LLMEngine:  # Karen: moteur haut niveau
    """Coordonne la génération sur un ou plusieurs GPU."""  # Karen: résumé fonctionnel

    def __init__(self, model, **kwargs):
        """Initialise le moteur et les processus de calcul.

        Parameters
        ----------
        model : str
            Chemin vers les poids du modèle.
        **kwargs : dict
            Autres options de :class:`Config` à surcharger.
        """

        config_fileds = {field.name for field in fields(Config)}  # Liste des attributs configurables
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fileds}  # Filtrage des paramètres
        config = Config(model, **config_kwargs)  # Création de la configuration globale
        self.ps = []  # Karen: liste de processus enfants
        self.events = []  # Karen: événements pour la syncro
        ctx = mp.get_context("spawn")  # Karen: contexte multiprocess compatible CUDA
        # Création des processus secondaires pour le parallélisme tensoriel
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # Signal pour communiquer avec le sous-processus
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()  # Lancement du sous-processus
            self.ps.append(process)  # Mémorisation du process
            self.events.append(event)  # Mémorisation de l'événement associé
        # Le processus principal instancie également un ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)  # Karen: runner principal
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)  # Karen: tokenizer
        config.eos = self.tokenizer.eos_token_id  # Karen: identifiant EOS
        self.scheduler = Scheduler(config)  # Karen: ordonnanceur des requêtes
        atexit.register(self.exit)  # Karen: nettoyage auto

    def exit(self):  # Karen: fermeture des ressources
        """Ferme proprement tous les processus lancés."""  # Karen: docstring
        self.model_runner.call("exit")  # Karen: signal d'arrêt
        del self.model_runner  # Karen: suppression de l'instance
        for p in self.ps:  # Karen: boucle sur les workers
            p.join()  # Karen: attente de la fin

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):  # Karen: nouvelle requête
        """Enfile une nouvelle séquence à générer."""  # Karen: description
        if isinstance(prompt, str):  # Karen: conversion éventuelle
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)  # Karen: création d'un objet séquence
        self.scheduler.add(seq)  # Karen: ajout au scheduler

    def step(self):  # Karen: exécute une itération complète
        """Effectue une itération de génération.

        Retourne les sorties finalisées et le nombre de tokens traités
        lors de cette étape (positif en préfill, négatif en décodage).
        """
        seqs, is_prefill = self.scheduler.schedule()  # Karen: obtention des séquences
        token_ids = self.model_runner.call("run", seqs, is_prefill)  # Karen: appel au runner
        self.scheduler.postprocess(seqs, token_ids)  # Karen: mise à jour des états
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]  # Karen: sorties prêtes
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)  # Karen: comptage des tokens
        return outputs, num_tokens  # Karen: renvoi des résultats

    def is_finished(self):  # Karen: statut de complétude
        """Indique s'il reste des séquences en cours."""  # Karen: docstring
        return self.scheduler.is_finished()  # Karen: délégation au scheduler

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:  # Karen: méthode de haut niveau
        """Génère les complétions pour une liste de prompts."""  # Karen: docstring
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)  # Karen: barre de progression
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)  # Karen: homogénéisation
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)  # Karen: enregistrement
        outputs = {}
        prefill_throughput = decode_throughput = 0.  # Karen: stats
        while not self.is_finished():
            t = perf_counter()  # Karen: début chrono
            output, num_tokens = self.step()  # Karen: appel step
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids  # Karen: stockage résultat
                if use_tqdm:
                    pbar.update(1)  # Karen: update barre
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]  # Karen: tri
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs  # Karen: liste de dictionnaires
