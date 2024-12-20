import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Charger les données depuis un fichier JSONL
def load_data(file_path):
    instructions = []
    responses = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            instructions.append(obj["instruction"])
            responses.append(obj["response"])
    return instructions, responses

# Créer un modèle de vecteur TF-IDF
def create_vectorizer(instructions):
    vectorizer = TfidfVectorizer()
    instruction_vectors = vectorizer.fit_transform(instructions)
    return vectorizer, instruction_vectors

# Initialiser le modèle GPT-2 pour la génération de réponses
@st.cache_resource
def initialize_gpt2():
    return pipeline("text-generation", model="gpt2")

# Fonction pour trouver la réponse la plus proche
def find_best_match(user_instruction, vectorizer, instruction_vectors, responses, gpt2_model):
    user_vector = vectorizer.transform([user_instruction])
    similarities = cosine_similarity(user_vector, instruction_vectors)
    best_match_index = similarities.argmax()
    best_similarity_score = similarities[0, best_match_index]

    # Réponses génériques
    generic_responses = {
        "mot de passe": "Un mot de passe robuste doit contenir au moins 12 caractères, avec des lettres majuscules, minuscules, chiffres et symboles.",
        "sécurité": "Pour assurer votre sécurité, activez l'authentification à deux facteurs et utilisez un gestionnaire de mots de passe.",
        "cybersécurité": "La cybersécurité implique la protection des systèmes, réseaux et programmes contre les cyberattaques.",
        "cloud": "Le cloud permet de stocker des données et d'accéder à des services en ligne à tout moment et depuis n'importe où.",
        "excel": "Pour sauvegarder un fichier Excel, utilisez la commande 'Fichier > Enregistrer sous'.",
        "vpn": "Un VPN (Virtual Private Network) est un outil qui vous permet de sécuriser votre connexion en cryptant vos données et en masquant votre adresse IP."
    }

    for keyword, response in generic_responses.items():
        if keyword in user_instruction.lower():
            return response

    # Si la similarité est faible, utiliser GPT-2 pour générer une réponse
    if best_similarity_score < 0.6:
        try:
            gpt2_response = gpt2_model(user_instruction, max_length=30, num_return_sequences=1, truncation=True)
            generated_text = gpt2_response[0]["generated_text"].strip()
            if len(generated_text.split()) < 5 or not generated_text.isascii() or "<|endoftext|>" in generated_text:
                return "Je ne suis pas sûr de comprendre votre question, mais essayez de reformuler."
            return generated_text
        except Exception as e:
            return "Je ne peux pas répondre à cette question pour le moment. Essayez une autre question."
    else:
        return responses[best_match_index]

# Application Streamlit
def main():
    st.title("Chatbot avec Streamlit")
    st.write("Posez votre question ci-dessous :")

    # Charger les données JSONL
    file_path = "Projet.jsonl"  # Remplacez par le chemin vers votre fichier JSONL
    instructions, responses = load_data(file_path)
    vectorizer, instruction_vectors = create_vectorizer(instructions)

    # Initialiser GPT-2
    gpt2_model = initialize_gpt2()

    # Entrée utilisateur
    user_instruction = st.text_input("Votre question :", "")

    # Afficher la réponse
    if st.button("Obtenir une réponse"):
        if user_instruction.strip():
            response = find_best_match(user_instruction, vectorizer, instruction_vectors, responses, gpt2_model)
            st.write(f"*Chatbot :* {response}")
        else:
            st.warning("Veuillez entrer une question.")

if __name__ == "__main__":
     main()