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

from nanovllm.config import Config  # Explanation: contient tous les paramètres nécessaires à l'initialisation
from nanovllm.sampling_params import SamplingParams  # Explanation: objet utilisé pour contrôler l'échantillonnage
from nanovllm.engine.sequence import Sequence  # Explanation: représente une requête et son état courant
from nanovllm.engine.scheduler import Scheduler  # Explanation: organise l'ordre de passage des séquences
from nanovllm.engine.model_runner import ModelRunner  # Explanation: entité responsable de l'exécution du modèle sur GPU


class LLMEngine:  # Explanation: composant maître qui gère plusieurs GPU et le flux des séquences
    """Coordonne la génération sur un ou plusieurs GPU."""

    def __init__(self, model, **kwargs):
        """Initialise le moteur et les processus de calcul.

        Parameters
        ----------
        model : str
            Chemin vers les poids du modèle.
        **kwargs : dict
            Autres options de :class:`Config` à surcharger.
        """

        config_fileds = {field.name for field in fields(Config)}  # Explanation: ensemble des champs autorisés dans la configuration
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fileds}  # Explanation: ne garde que les clés reconnues par `Config`
        config = Config(model, **config_kwargs)  # Explanation: construit l'objet de configuration à partir des valeurs fournies
        self.ps = []  # Explanation: stocke les processus enfants lancés sur d'autres GPU
        self.events = []  # Explanation: événements utilisés pour synchroniser les sous-processus
        ctx = mp.get_context("spawn")  # Explanation: contexte de multiprocessing adapté à CUDA
        # Création des processus secondaires pour le parallélisme tensoriel
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # Explanation: objet de synchronisation partagé avec ce worker
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()  # Explanation: démarre réellement le worker GPU
            self.ps.append(process)  # Explanation: conserve la référence pour pouvoir le rejoindre plus tard
            self.events.append(event)  # Explanation: permet au processus principal d'envoyer des ordres
        # Le processus principal instancie également un ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)  # Explanation: instancie le `ModelRunner` du processus maître
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)  # Explanation: charge le tokenizer associé au modèle
        config.eos = self.tokenizer.eos_token_id  # Explanation: enregistre l'identifiant du token de fin dans la configuration
        self.scheduler = Scheduler(config)  # Explanation: composant chargé de décider quand exécuter chaque séquence
        atexit.register(self.exit)  # Explanation: garantit l'arrêt propre des processus si le programme se termine

    def exit(self):  # Explanation: méthode appelée lors de la fermeture pour nettoyer les GPU
        """Ferme proprement tous les processus lancés."""
        self.model_runner.call("exit")  # Explanation: demande au ModelRunner principal de libérer ses ressources
        del self.model_runner  # Explanation: supprime l'instance pour forcer la libération mémoire
        for p in self.ps:  # Explanation: parcourt tous les processus secondaires
            p.join()  # Explanation: attend qu'ils se terminent avant de quitter

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):  # Explanation: enregistre une requête dans la file d'attente
        """Enfile une nouvelle séquence à générer."""
        if isinstance(prompt, str):  # Explanation: si `prompt` est du texte brut, on le tokenise
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)  # Explanation: encapsule le prompt et les paramètres dans une `Sequence`
        self.scheduler.add(seq)  # Explanation: place la séquence dans la file d'attente

    def step(self):  # Explanation: effectue une étape de génération pour toutes les séquences sélectionnées
        """Effectue une itération de génération.

        Retourne les sorties finalisées et le nombre de tokens traités
        lors de cette étape (positif en préfill, négatif en décodage).
        """
        seqs, is_prefill = self.scheduler.schedule()  # Explanation: récupère les séquences prêtes à être traitées et indique le mode
        token_ids = self.model_runner.call("run", seqs, is_prefill)  # Explanation: envoie la liste au `ModelRunner` qui renvoie les nouveaux tokens
        self.scheduler.postprocess(seqs, token_ids)  # Explanation: met à jour l'état de chaque séquence après l'exécution
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]  # Explanation: récupère les résultats terminés
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)  # Explanation: compte les tokens pour calculer le throughput
        return outputs, num_tokens  # Explanation: renvoie à la boucle principale les nouvelles sorties et le nombre de tokens

    def is_finished(self):  # Explanation: permet de savoir si toutes les séquences ont été traitées
        """Indique s'il reste des séquences en cours."""
        return self.scheduler.is_finished()  # Explanation: interroge le scheduler pour obtenir l'information

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:  # Explanation: interface complète pour générer plusieurs textes en parallèle
        """Génère les complétions pour une liste de prompts."""
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)  # Explanation: affichage d'une barre de progression dans le terminal
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)  # Explanation: duplique les paramètres pour chaque prompt
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)  # Explanation: enfile chaque paire prompt/params dans la file d'attente
        outputs = {}
        prefill_throughput = decode_throughput = 0.  # Explanation: suivra les performances en pré-remplissage et en décodage
        while not self.is_finished():
            t = perf_counter()  # Explanation: démarre le chronomètre pour cette étape
            output, num_tokens = self.step()  # Explanation: exécute effectivement le modèle
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
                outputs[seq_id] = token_ids  # Explanation: mémorise les tokens générés de chaque séquence finalisée
                if use_tqdm:
                    pbar.update(1)  # Explanation: incrémente la barre d'un cran
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]  # Explanation: remet les réponses dans l'ordre original
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs  # Explanation: renvoie une liste de dictionnaires contenant texte et identifiants
