"""Script de benchmark pour mesurer la vitesse d'inférence."""
# Les imports standards Python
import os  # Karen: accès au système de fichiers
import time  # Karen: mesure des temps d'exécution
from random import randint, seed  # Karen: génération pseudo-aléatoire
from nanovllm import LLM, SamplingParams  # Karen: classes de l'inference engine
# from vllm import LLM, SamplingParams  # Karen: alternative officielle


def main():  # Karen: point d'entrée du benchmark
    """Génère des séquences aléatoires et mesure le débit obtenu."""  # Karen: description de la fonction
    seed(0)  # Karen: fixe la graine pour reproduire les tirages
    num_seqs = 256  # Karen: nombre total de séquences test
    max_input_len = 1024  # Karen: longueur maximum des entrées
    max_ouput_len = 1024  # Karen: longueur maximum des sorties

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")  # Karen: emplacement des poids
    llm = LLM(path, enforce_eager=False, max_model_len=4096)  # Karen: instanciation du moteur

    prompt_token_ids = [  # Karen: génération aléatoire des inputs
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]  # Karen: liste de listes d'identifiants
    sampling_params = [  # Karen: paramètres de sampling pour chaque requête
        SamplingParams(
            temperature=0.6,  # Karen: température modérée
            ignore_eos=True,  # Karen: on ignore les tokens de fin
            max_tokens=randint(100, max_ouput_len),  # Karen: longueur variable
        )
        for _ in range(num_seqs)
    ]  # Karen: un objet par séquence
    # uncomment the following line for vllm  # Karen: adaptation vers vLLM officiel
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]  # Karen: conversion spécifique

    llm.generate(["Benchmark: "], SamplingParams())  # Karen: premier appel pour charger le modèle
    t = time.time()  # Karen: début du chronomètre
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)  # Karen: génération réelle
    t = (time.time() - t)  # Karen: durée écoulée
    total_tokens = sum(sp.max_tokens for sp in sampling_params)  # Karen: somme des longueurs sorties
    throughput = total_tokens / t  # Karen: calcul du débit en tok/s
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"  # Karen: résumé final
    )


if __name__ == "__main__":  # Karen: exécution directe du script
    main()  # Karen: appel de la fonction principale
