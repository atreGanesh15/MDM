# chatbot.py
import numpy as np
import joblib

# ============================
# 1. Load trained artifacts
# ============================

model = joblib.load("symptom_checker_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
symptom_names = joblib.load("symptom_names.joblib")  # list of strings


# ============================
# 2. Helper functions
# ============================

def symptoms_to_vector(user_symptoms, symptom_names):
    """
    Convert a list of symptom names to a 0/1 vector
    matching the order in symptom_names.
    """
    vector = np.zeros(len(symptom_names), dtype=np.int32)

    for s in user_symptoms:
        s_clean = s.strip().lower()
        # match ignoring case / extra spaces
        found = False
        for idx, col_name in enumerate(symptom_names):
            if s_clean == col_name.strip().lower():
                vector[idx] = 1
                found = True
                break

        if not found:
            print(f"⚠️  Warning: symptom '{s}' not found in trained symptom list.")

    return vector.reshape(1, -1)


def predict_disease(user_symptoms):
    """
    Takes a list of symptom strings, returns predicted disease name.
    """
    x_vec = symptoms_to_vector(user_symptoms, symptom_names)
    pred_encoded = model.predict(x_vec)[0]
    disease = label_encoder.inverse_transform([pred_encoded])[0]
    return disease


def get_advice_for_disease(disease):
    """
    Very simple rule-based advice.
    You can expand this dictionary as needed.
    """
    disease_lower = disease.lower()

    advice_db = {
        "common cold": (
            "It may be a common cold. Rest well, stay hydrated, "
            "and consider over-the-counter cold medicine if needed. "
            "If symptoms last more than a week or worsen, consult a doctor."
        ),
        "covid-19": (
            "Symptoms may indicate COVID-19. Self-isolate, wear a mask, "
            "and get tested as soon as possible. Seek medical help if "
            "you experience breathing difficulty, chest pain, or confusion."
        ),
        "migraine": (
            "This looks like a migraine. Rest in a dark, quiet room and "
            "avoid screen time. If headaches are severe or very frequent, "
            "visit a neurologist."
        ),
        "diabetes": (
            "These symptoms may be related to diabetes. It's important to "
            "consult a doctor for blood sugar tests and long-term management."
        ),
    }

    # default advice
    default_advice = (
        "Based on your symptoms, this condition may need medical attention. "
        "Please consult a healthcare professional for confirmation, tests, "
        "and proper treatment."
    )

    # try exact match first
    if disease_lower in advice_db:
        return advice_db[disease_lower]

    # otherwise fall back to generic advice
    return default_advice


# ============================
# 3. Simple chat loop
# ============================

def run_chatbot():
    print("👋 Hi, I'm your AI Symptom Checker Chatbot.")
    print("Type your symptoms separated by commas (e.g. 'fever, cough, headache').")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Take care! Goodbye. 👋")
            break

        if not user_input:
            print("Bot: Please enter at least one symptom.")
            continue

        # Parse symptoms
        user_symptoms = [s for s in user_input.split(",") if s.strip()]

        # Predict disease
        try:
            predicted_disease = predict_disease(user_symptoms)
        except Exception as e:
            print("Bot: Sorry, I couldn't process that. Please try again.")
            print(f"(Debug info: {e})")
            continue

        # Get advice
        advice = get_advice_for_disease(predicted_disease)

        print(f"\nBot: Based on your symptoms, you may have: **{predicted_disease}**")
        print("Bot:", advice)
        print("\n(⚠️ This is not a medical diagnosis. Always consult a doctor for confirmation.)\n")


if __name__ == "__main__":
    run_chatbot()
