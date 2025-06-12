import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load trained model
model = joblib.load(Path("artifacts/model_training/grid_search_model.joblib"))

st.title("🤖 Banking Product Recommender Chatbot")

# Create a chat container
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to extract structured features from a message (rule-based parser)
def parse_user_input(message):
    # Default values
    data = {
        "TransactionType": "Deposit",
        "ProductCategory": "Loan",
        "ProductSubcategory": "Standard",
        "BranchCity": "Seville",
        "Channel": "ATM",
        "Amount": 1000.0,
        "CreditCardFees": 0.0,
        "InsuranceFees": 0.0,
        "LatePaymentAmount": 0.0,
        "CustomerScore": 600,
        "MonthlyIncome": 2000.0,
        "Month": 5,
        "Year": 2025
    }

    # Parse keywords from message
    msg = message.lower()

    # Transaction types
    if "withdraw" in msg:
        data["TransactionType"] = "Withdrawal"
    elif "transfer" in msg:
        data["TransactionType"] = "Transfer"
    elif "card" in msg:
        data["TransactionType"] = "Card Payment"

    # Product category
    for cat in ["loan", "mortgage", "checking", "credit"]:
        if cat in msg:
            if cat == "credit":
                data["ProductCategory"] = "Credit Card"
            elif cat == "checking":
                data["ProductCategory"] = "Checking Account"
            else:
                data["ProductCategory"] = cat.capitalize()

    # Product subcategory
    for sub in ["gold", "standard", "platinum"]:
        if sub in msg:
            data["ProductSubcategory"] = sub.capitalize()

    # City
    for city in ["malaga", "murcia", "seville"]:
        if city in msg:
            data["BranchCity"] = city.capitalize()

    # Channel
    for ch in ["atm", "branch"]:
        if ch in msg:
            data["Channel"] = ch.capitalize()

    # Optional: extract amounts or income from numbers in message (advanced)

    return pd.DataFrame([data])

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
user_input = st.chat_input("Hi! What banking product are you looking for?")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Parse input and get prediction
    input_df = parse_user_input(user_input)
    recommendation = model.predict(input_df)[0]

    bot_response = f"Based on what you've told me, I recommend: **{recommendation}** 💡"
    
    # Save and display bot message
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
