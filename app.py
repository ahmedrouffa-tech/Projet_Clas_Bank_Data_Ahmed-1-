import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Configuration de la page
st.set_page_config(page_title="Prédiction Dépôt Bancaire", page_icon="🏦", layout="centered")
st.title("🏦 Prédiction de Souscription au Dépôt à Terme")
st.write("Cette application utilise un modèle XGBoost pour prédire si un client souscrira à un dépôt à terme.")

# 2. Chargement des ressources
@st.cache_resource
def load_resources():
    try:
        resources = joblib.load('list_items.joblib')
        return resources
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier de configuration : {e}")
        return None

resources = load_resources()

if resources:
    unique_values = resources['uniques']
    scaler = resources['scaler']
    model = resources['model']

    # Récupérer l'ordre exact des colonnes attendu par le scaler et le modèle
    # Le scaler a été entraîné sur 46 colonnes issues de pd.get_dummies
    expected_columns = [
        'age', 'duration', 'campaign', 'pdays', 'previous',
        'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
        'job_retired', 'job_self-employed', 'job_services', 'job_student',
        'job_technician', 'job_unemployed', 'job_unknown', 'marital_married',
        'marital_single', 'marital_unknown', 'education_basic.6y',
        'education_basic.9y', 'education_high.school', 'education_illiterate',
        'education_professional.course', 'education_university.degree',
        'education_unknown', 'housing_unknown', 'housing_yes', 'loan_unknown',
        'loan_yes', 'contact_telephone', 'month_aug', 'month_dec', 'month_jul',
        'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct',
        'month_sep', 'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue',
        'day_of_week_wed', 'poutcome_nonexistent', 'poutcome_success'
    ]

    # 3. Formulaire de saisie utilisateur
    st.header("Informations du Client")
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

    # 4. Traitement et Prédiction
    if st.button("Prédire la souscription", type="primary"):
        # Créer le DataFrame de base avec les colonnes numériques et catégorielles d'origine
        input_df = pd.DataFrame({
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

        # Appliquer le One-Hot Encoding comme sur les données de base
        input_encoded = pd.get_dummies(input_df)

        # Aligner les colonnes avec les 46 colonnes de l'entraînement
        # Si une colonne de One-Hot Encoding est absente, on l'ajoute avec la valeur 0
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Réordonner les colonnes exactement comme attendu par le scaler et le modèle
        input_final = input_encoded[expected_columns]

        # Standardisation
        input_scaled = scaler.transform(input_final)

        # Prédiction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Résultat de la prédiction :")
        if prediction == 1:
            st.success(f"🎉 Le client est susceptible de souscrire ! (Probabilité : {probability:.2%})")
        else:
            st.error(f"❌ Le client n'est pas susceptible de souscrire. (Probabilité de refus : {1 - probability:.2%})")
