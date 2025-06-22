import os
"""Exemple minimal d'utilisation de Nano-vLLM."""

# Importation des classes nécessaires

from nanovllm import LLM, SamplingParams  # Classe du moteur et paramètres
from transformers import AutoTokenizer  # Tokeniseur standard HuggingFace


def main():
    """Lance une courte génération pour démonstration."""
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")  # Chemin du modèle
    tokenizer = AutoTokenizer.from_pretrained(path)  # Chargement du tokenizer
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)  # Moteur LLM en mode eager

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)  # Paramètres globaux
    prompts = [
        "introduce yourself",  # Premier prompt
        "list all prime numbers within 100",  # Deuxième prompt
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],  # Contenu du message utilisateur
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)  # Lancement de la génération

    for prompt, output in zip(prompts, outputs):
        print("\n")  # Séparation visuelle
        print(f"Prompt: {prompt!r}")  # Affichage du prompt
        print(f"Completion: {output['text']!r}")  # Affichage de la réponse


if __name__ == "__main__":
    main()
