"""Script de benchmark pour mesurer la vitesse d'inférence."""
# Les imports standards Python
import os
import time
from random import randint, seed  # Génération aléatoire de tokens
from nanovllm import LLM, SamplingParams  # API Nano-vLLM pour l'inférence
# from vllm import LLM, SamplingParams  # Optionnel : version officielle


def main():
    """Génère des séquences aléatoires et mesure le débit obtenu."""
    seed(0)  # Graine fixe pour la reproductibilité
    num_seqs = 256  # Nombre total de séquences testées
    max_input_len = 1024  # Taille maximale des entrées
    max_ouput_len = 1024  # Taille maximale des sorties

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")  # Chemin du modèle
    llm = LLM(path, enforce_eager=False, max_model_len=4096)  # Instanciation du moteur

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]  # Données d'entrée simulées
    sampling_params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=randint(100, max_ouput_len),
        )
        for _ in range(num_seqs)
    ]  # Paramètres de génération propres à chaque séquence
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())  # Appel initial pour charger le modèle
    t = time.time()  # Début de la mesure
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)  # Durée totale
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
