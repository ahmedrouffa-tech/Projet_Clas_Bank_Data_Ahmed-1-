from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# 1. CONFIGURATION DE L'INTERFACE
# ==========================================
st.set_page_config(
    page_title="Prédiction Dépôt Bancaire",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 Prédiction de Souscription au Dépôt à Terme")
st.write(
    "Cette application utilise un modèle de machine learning pour estimer "
    "si un client souscrira à un dépôt à terme."
)


# ==========================================
# 2. CHARGEMENT DES RESSOURCES
# ==========================================
MODEL_PATH = Path(__file__).resolve().parent / "list_items.joblib"


@st.cache_resource
def load_resources():
    """Charge les ressources du modèle depuis le fichier joblib."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {MODEL_PATH}")

    resources = joblib.load(MODEL_PATH)

    if not isinstance(resources, dict):
        raise TypeError("Le fichier de ressources doit contenir un dictionnaire.")

    return resources


try:
    resources = load_resources()
except Exception as error:
    st.error(f"Impossible de charger les ressources du modèle : {error}")
    st.stop()

encoders = resources.get("encoders") or {}
unique_values = resources.get("uniques") or {}
scaler = resources.get("scaler")
model = resources.get("model")

if scaler is None or model is None:
    st.error(
        "Les ressources du modèle sont incomplètes. "
        "Vérifiez le contenu de `list_items.joblib`."
    )
    st.stop()


# ==========================================
# 3. OPTIONS ET VALEURS PAR DÉFAUT
# ==========================================
CATEGORICAL_FIELDS = [
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

NUMERIC_DEFAULTS = {
    "age": 35,
    "duration": 150,
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
}

CATEGORY_FALLBACKS = {
    "job": "admin.",
    "marital": "married",
    "education": "secondary",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "thu",
    "poutcome": "unknown",
}


def get_options(field):
    """Récupère les options d'une catégorie depuis les ressources."""
    values = unique_values.get(field)

    if values is None:
        encoder = encoders.get(field)
        values = getattr(encoder, "classes_", None)

    if values is None:
        return []

    if isinstance(values, pd.Series):
        values = values.tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    elif isinstance(values, set):
        values = sorted(values)
    elif not isinstance(values, (list, tuple)):
        values = [values]

    # Élimine les valeurs manquantes sans réordonner la liste.
    options = []
    for value in values:
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        if value not in options:
            options.append(value)

    return options


OPTIONS = {field: get_options(field) for field in CATEGORICAL_FIELDS}

# Les sélecteurs ne peuvent pas fonctionner sans options.
empty_fields = [field for field, options in OPTIONS.items() if not options]
if empty_fields:
    st.error(
        "Options manquantes dans `list_items.joblib` pour : "
        + ", ".join(empty_fields)
    )
    st.stop()

DEFAULTS = dict(NUMERIC_DEFAULTS)

for field in CATEGORICAL_FIELDS:
    options = OPTIONS[field]
    preferred_value = CATEGORY_FALLBACKS[field]

    # Utilise la valeur habituelle si elle existe, sinon la première option disponible.
    DEFAULTS[field] = (
        preferred_value if preferred_value in options else options[0]
    )


def reset_form():
    """Restaure toutes les valeurs par défaut du formulaire."""
    for field, value in DEFAULTS.items():
        st.session_state[field] = value


# Initialise l'état des widgets et corrige les anciennes valeurs invalides.
for field, default_value in DEFAULTS.items():
    if field not in st.session_state:
        st.session_state[field] = default_value
    elif field in CATEGORICAL_FIELDS and st.session_state[field] not in OPTIONS[field]:
        st.session_state[field] = default_value


# ==========================================
# 4. PRÉPARATION DES DONNÉES
# ==========================================
FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


def prepare_input_data(form_values):
    """Construit le DataFrame et encode les catégories."""
    input_data = pd.DataFrame(
        [{field: form_values[field] for field in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )

    for field, encoder in encoders.items():
        if field not in input_data.columns:
            continue

        value = input_data.at[0, field]
        classes = getattr(encoder, "classes_", None)

        if classes is not None and value not in classes:
            raise ValueError(
                f"La valeur « {value} » du champ « {field} » "
                "n'est pas connue de l'encodeur du modèle."
            )

        try:
            input_data[field] = encoder.transform(input_data[field])
        except Exception as error:
            raise ValueError(
                f"Impossible d'encoder le champ « {field} » : {error}"
            ) from error

    numeric_fields = ["age", "duration", "campaign", "pdays", "previous"]
    for field in numeric_fields:
        input_data[field] = pd.to_numeric(input_data[field], errors="coerce")

    if input_data.isnull().any().any():
        raise ValueError("Une ou plusieurs valeurs sont invalides ou manquantes.")

    # Si le scaler connaît les noms de colonnes, respecte leur ordre d'entraînement.
    scaler_columns = getattr(scaler, "feature_names_in_", None)
    if scaler_columns is not None:
        expected_columns = list(scaler_columns)
        if set(expected_columns) != set(input_data.columns):
            raise ValueError(
                "Les colonnes fournies ne correspondent pas aux colonnes attendues "
                "par le scaler."
            )
        input_data = input_data[expected_columns]

    return input_data


# ==========================================
# 5. FORMULAIRE
# ==========================================
st.header("Informations du client")

# Ce bouton est placé hors du formulaire pour pouvoir réinitialiser
# l'état des widgets sans conflit avec leur création.
st.button(
    "↩️ Réinitialiser",
    on_click=reset_form,
    help="Rétablir les valeurs par défaut du formulaire.",
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.number_input(
            "Âge",
            min_value=18,
            max_value=100,
            key="age",
        )
        st.selectbox("Profession", OPTIONS["job"], key="job")
        st.selectbox("Statut marital", OPTIONS["marital"], key="marital")
        st.selectbox("Éducation", OPTIONS["education"], key="education")
        st.selectbox("Prêt immobilier ?", OPTIONS["housing"], key="housing")
        st.selectbox("Prêt personnel ?", OPTIONS["loan"], key="loan")
        st.selectbox("Moyen de contact", OPTIONS["contact"], key="contact")

    with col2:
        st.selectbox(
            "Mois du dernier contact",
            OPTIONS["month"],
            key="month",
        )
        st.selectbox(
            "Jour de la semaine",
            OPTIONS["day_of_week"],
            key="day_of_week",
        )
        st.number_input(
            "Durée du dernier appel (secondes)",
            min_value=0,
            max_value=5000,
            key="duration",
        )
        st.number_input(
            "Nombre de contacts durant cette campagne",
            min_value=1,
            max_value=50,
            key="campaign",
        )
        st.number_input(
            "Jours écoulés depuis le contact précédent (999 = jamais)",
            min_value=0,
            max_value=999,
            key="pdays",
        )
        st.number_input(
            "Nombre de contacts précédents",
            min_value=0,
            max_value=10,
            key="previous",
        )
        st.selectbox(
            "Résultat de la campagne précédente",
            OPTIONS["poutcome"],
            key="poutcome",
        )

    submitted = st.form_submit_button(
        "Prédire la souscription",
        type="primary",
    )


# ==========================================
# 6. PRÉDICTION
# ==========================================
if submitted:
    form_values = {
        field: st.session_state[field]
        for field in FEATURE_COLUMNS
    }

    try:
        input_data = prepare_input_data(form_values)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.subheader("Résultat de la prédiction")

        if prediction == 1:
            message = "Le client est susceptible de souscrire."
        else:
            message = "Le client n'est pas susceptible de souscrire."

        # Affiche la probabilité si le modèle la fournit.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)[0]
            classes = list(getattr(model, "classes_", []))

            positive_index = next(
                (
                    index
                    for index, value in enumerate(classes)
                    if value == 1 or value == "1"
                ),
                None,
            )

            if positive_index is not None:
                probability = float(probabilities[positive_index])

                if prediction == 1:
                    st.success(f"🎉 {message} Probabilité : {probability:.2%}")
                else:
                    st.error(
                        f"❌ {message} "
                        f"Probabilité de souscription : {probability:.2%}"
                    )
            else:
                st.info(message)
        elif prediction == 1:
            st.success(f"🎉 {message}")
        else:
            st.error(f"❌ {message}")

    except Exception as error:
        st.error(f"Erreur lors de la prédiction : {error}")
