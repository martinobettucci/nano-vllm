import os  # Karen: nécessaire pour chemin utilisateur
"""Exemple minimal d'utilisation de Nano-vLLM."""  # Karen: description en une ligne

# Importation des classes nécessaires

from nanovllm import LLM, SamplingParams  # Karen: moteur et paramètres
from transformers import AutoTokenizer  # Karen: tokeniseur HuggingFace


def main():  # Karen: fonction principale
    """Lance une courte génération pour démonstration."""  # Karen: résumé
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")  # Karen: dossier modèle
    tokenizer = AutoTokenizer.from_pretrained(path)  # Karen: ouverture du tokenizer
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)  # Karen: moteur en mode eager

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)  # Karen: réglages de sampling
    prompts = [
        "introduce yourself",  # Karen: premier prompt
        "list all prime numbers within 100",  # Karen: deuxième prompt
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],  # Karen: message utilisateur
            tokenize=False,  # Karen: pas de tokenisation ici
            add_generation_prompt=True,  # Karen: ajoute le séparateur
            enable_thinking=True,  # Karen: active la réflexion
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)  # Karen: appel au moteur

    for prompt, output in zip(prompts, outputs):  # Karen: boucle d'affichage
        print("\n")  # Karen: saut de ligne
        print(f"Prompt: {prompt!r}")  # Karen: on rappelle le prompt
        print(f"Completion: {output['text']!r}")  # Karen: on montre la complétion


if __name__ == "__main__":  # Karen: exécution directe
    main()  # Karen: appel de main
