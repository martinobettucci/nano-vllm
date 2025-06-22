import os  # Explanation: utilisé pour résoudre le chemin d'accès au modèle
"""Exemple minimal d'utilisation de Nano-vLLM."""  # Explanation: résumé en une ligne de l'objectif du script

# Importation des classes nécessaires

from nanovllm import LLM, SamplingParams  # Explanation: `LLM` est le moteur d'inférence et `SamplingParams` définit la stratégie de génération
from transformers import AutoTokenizer  # Explanation: charge le tokenizer compatible avec le modèle


def main():  # Explanation: point d'entrée qui prépare le modèle et lance la génération
    """Lance une courte génération pour démonstration."""
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")  # Explanation: dossier contenant les poids du modèle ; `expanduser` remplace `~`
    tokenizer = AutoTokenizer.from_pretrained(path)  # Explanation: charge le tokenizer correspondant aux poids
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)  # Explanation: instancie le moteur en mode eager pour simplifier le débogage

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)  # Explanation: configuration utilisée pour toutes les requêtes
    prompts = [
        "introduce yourself",  # Explanation: premier message envoyé au modèle
        "list all prime numbers within 100",  # Explanation: second exemple de question
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],  # Explanation: structure de conversation attendue par le modèle
            tokenize=False,  # Explanation: on laisse le moteur `LLM` gérer la tokenisation
            add_generation_prompt=True,  # Explanation: ajoute le séparateur de début de réponse
            enable_thinking=True,  # Explanation: active la balise spéciale qui pousse le modèle à raisonner
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)  # Explanation: envoie la liste de prompts au moteur et récupère les réponses

    for prompt, output in zip(prompts, outputs):  # Explanation: on parcourt les couples (prompt, réponse) pour les afficher
        print("\n")  # Explanation: insère une ligne vide pour la lisibilité
        print(f"Prompt: {prompt!r}")  # Explanation: affiche le texte d'origine
        print(f"Completion: {output['text']!r}")  # Explanation: affiche le texte généré par le modèle


if __name__ == "__main__":  # Explanation: ne se lance que si ce fichier est exécuté comme script
    main()  # Explanation: déclenche la génération de l'exemple
