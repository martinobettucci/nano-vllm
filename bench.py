"""Script de benchmark pour mesurer la vitesse d'inférence."""
# Les imports standards Python
import os  # Explanation: fournit les fonctions pour manipuler les chemins et l'environnement
import time  # Explanation: mesure la durée d'exécution du benchmark
from random import randint, seed  # Explanation: `randint` produit des entiers aléatoires et `seed` fige la suite générée
from nanovllm import LLM, SamplingParams  # Explanation: `LLM` exécute le modèle et `SamplingParams` décrit la génération
# from vllm import LLM, SamplingParams  # Explanation: référence vers l'implémentation officielle pour comparaison


def main():  # Explanation: fonction centrale qui lance tout le processus de mesure
    """Génère des séquences aléatoires et mesure le débit obtenu."""
    seed(0)  # Explanation: fixe la graine pour que chaque exécution produise les mêmes données
    num_seqs = 256  # Explanation: nombre total de séquences évaluées
    max_input_len = 1024  # Explanation: longueur maximale de l'entrée aléatoire
    max_ouput_len = 1024  # Explanation: borne haute pour la sortie générée

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")  # Explanation: chemin vers le modèle local, le ~ est résolu automatiquement
    llm = LLM(path, enforce_eager=False, max_model_len=4096)  # Explanation: création du moteur `LLM` avec exécution optimisée et longueur max autorisée

    prompt_token_ids = [  # Explanation: chaque élément sera une liste de tokens d'entrée générés aléatoirement
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]  # Explanation: structure finale contenant les prompts déjà tokenisés
    sampling_params = [  # Explanation: options de génération individuelles pour chaque séquence
        SamplingParams(
            temperature=0.6,  # Explanation: valeur de la température pour la distribution
            ignore_eos=True,  # Explanation: permet de continuer même si le modèle propose le token de fin
            max_tokens=randint(100, max_ouput_len),  # Explanation: nombre de tokens générés aléatoirement pour la réponse
        )
        for _ in range(num_seqs)
    ]  # Explanation: création d'autant de `SamplingParams` que de séquences
    # uncomment the following line for vllm  # Explanation: utile uniquement si on souhaite comparer avec l'API vLLM
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]  # Explanation: mise en forme spécifique à vLLM

    llm.generate(["Benchmark: "], SamplingParams())  # Explanation: premier appel à vide pour initialiser le modèle en mémoire
    t = time.time()  # Explanation: instant de départ du chrono
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)  # Explanation: génération des sorties pour toutes les séquences
    t = (time.time() - t)  # Explanation: temps total écoulé pour produire toutes les sorties
    total_tokens = sum(sp.max_tokens for sp in sampling_params)  # Explanation: nombre de tokens effectivement produits durant le test
    throughput = total_tokens / t  # Explanation: calcul du débit en tokens par seconde
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"  # Explanation: affiche les statistiques de performance
    )


if __name__ == "__main__":  # Explanation: exécute le benchmark uniquement si ce fichier est lancé directement
    main()  # Explanation: démarre la fonction principale
