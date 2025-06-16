import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# === Load trained model and label encoder ===
model_path = Path("artifacts/retrained_model/fin_model.joblib")
model = joblib.load(model_path)
labelencoder = joblib.load(Path("artifacts/retrained_model/labelencorder.joblib"))

st.title("🤖 Banking Product Recommender Chatbot")
st.markdown("Ask me what kind of banking product suits your situation, and I’ll suggest one for you!")

# === Setup chat history ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === Function to parse user input ===
def parse_user_input(message: str) -> pd.DataFrame:
    msg = message.lower()

    # Default feature values
    data = {
        "monthlyincome": 2500.0,
        "productcategory": "Loan",
        "most_used_channel": "ATM",
        "productsubcategory": "Standard",
        "amount": 1000.0,
    }

    # Extract features based on keywords
    for cat in ["loan", "mortgage", "credit", "savings", "checking"]:
        if cat in msg:
            data["productcategory"] = "Credit Card" if cat == "credit" else cat.capitalize()

    for sub in ["gold", "platinum", "standard"]:
        if sub in msg:
            data["productsubcategory"] = sub.capitalize()

    for ch in ["atm", "branch", "mobile", "online"]:
        if ch in msg:
            data["most_used_channel"] = ch.capitalize()

    # Extract numeric values for income or amount
    for word in msg.split():
        if word.replace('.', '', 1).isdigit():
            val = float(word)
            if "income" in msg or "salary" in msg:
                data["monthlyincome"] = val
            else:
                data["amount"] = val

    return pd.DataFrame([data])

# === Display chat history ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === Get user input ===
user_input = st.chat_input("Tell me about your needs...", key="chat_input")

if user_input:
    # Save and show user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        # Parse message to features
        input_df = parse_user_input(user_input)

        # Make prediction
        encoded_prediction = model.predict(input_df)[0]

        # Decode prediction to original label
        decoded_prediction = labelencoder.inverse_transform([encoded_prediction])[0]

        response = f"Based on your needs, I recommend: **{decoded_prediction}** 💡"

    except Exception as e:
        response = "⚠️ Sorry, I couldn't understand that. Please provide more details about your income, product type, or channel used."
        st.error(str(e))  # Debugging output

    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
