import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========================================== 
# 1. CONFIGURATION DE L'INTERFACE STREAMLIT
# ==========================================
st.set_page_config(page_title="Prédiction Dépôt Bancaire", page_icon="🏦", layout="centered")
st.title("🏦 Prédiction de Souscription au Dépôt à Terme")
st.write("Cette application utilise un modèle XGBoost pour prédire si un client souscrira à un dépôt à terme.")

# ==========================================
# 2. CHARGEMENT DES RESSOURCES (CACHE)
# ==========================================
# On utilise @st.cache_resource pour éviter de recharger les fichiers lourds à chaque interaction de l'utilisateur
@st.cache_resource
def load_resources():
    try:
        # list_items.joblib contient : le scaler, les encodeurs, le modèle et les valeurs uniques pour les listes déroulantes
        resources = joblib.load('list_items.joblib')
        return resources
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles et encodeurs : {e}")
        return None

resources = load_resources()

if resources:
    # Extraction des différents objets nécessaires au pipeline de prédiction
    encoders = resources['encoders']
    unique_values = resources['uniques']
    scaler = resources['scaler']
    model = resources['model']

    # ==========================================
# 3. FORMULAIRE DE SAISIE UTILISATEUR
# ==========================================
    st.header("Informations du Client")

    # Division de l'écran en deux colonnes pour une meilleure disposition visuelle
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=35)
        job = st.selectbox("Profession", unique_values['job'])
        marital = st.selectbox("Statut Marital", unique_values['marital'])
        education = st.selectbox("Éducation", unique_values['education'])
        housing = st.selectbox("Prêt Immobilier ?", unique_values['housing'])
        loan = st.selectbox("Prêt Personnel ?", unique_values['loan'])
        contact = st.selectbox("Moyen de Contact", unique_values['contact'])

    with col2:
        month = st.selectbox("Mois du dernier contact", unique_values['month'])
        day_of_week = st.selectbox("Jour de la semaine", unique_values['day_of_week'])
        duration = st.number_input("Durée du dernier appel (sec)", min_value=0, max_value=5000, value=150)
        campaign = st.number_input("Nombre de contacts durant cette campagne", min_value=1, max_value=50, value=1)
        pdays = st.number_input("Jours écoulés depuis contact précédent (999 = jamais)", min_value=0, max_value=999, value=999)
        previous = st.number_input("Nombre de contacts précédents", min_value=0, max_value=10, value=0)
        poutcome = st.selectbox("Résultat campagne précédente", unique_values['poutcome'])

    # ==========================================
# 4. CRÉATION DU DATAFRAME ET ENCODAGE
# ==========================================
    # Regroupement des saisies dans un DataFrame pandas au format attendu par le modèle
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    # Encodage des variables catégorielles à l'aide des LabelEncoders ajustés lors de la phase d'entraînement
    for col in encoders:
        le = encoders[col]
        input_data[col] = le.transform(input_data[col])

    # ==========================================
# 5. EXÉCUTION DE LA PRÉDICTION
# ==========================================
    if st.button("Prédire la souscription", type="primary"):
        # Étape indispensable : Appliquer la même normalisation que sur l'ensemble d'entraînement
        input_scaled = scaler.transform(input_data)

        # Récupération de la classe prédite (0 ou 1) et de la probabilité associée
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Résultat de la prédiction :")
        if prediction == 1:
            st.success(f"🎉 Le client est susceptible de souscrire ! (Probabilité : {probability:.2%})")
        else: 
            st.error(f"❌ Le client n'est pas susceptible de souscrire. (Probabilité de refus : {1 - probability:.2%})")
