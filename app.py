import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from openai import OpenAI
import plotly.express as px

# 1. SECURE CREDENTIALS (ALWAYS ON MODE)
# Uses Streamlit Secrets so you never have to type keys again.
try:
    fred_key = st.secrets["FRED_KEY"]
    ai_key = st.secrets["AI_KEY"]
except:
    st.error("🔑 Keys not found! Go to Settings -> Secrets in Streamlit and add them.")
    st.stop()

# 2. BRANDING & HEADER
st.title("📈 TCPI: Consumer Price Index")
st.caption(f"🚀 TCPI Startup | Founder: [Your Name] | Status: Live & Automated")
st.markdown("### *AI-Powered Economic Forecasting for the Next Generation*")

if fred_key and ai_key:
    # 3. DATA ENGINE: Fetch Live Inflation Data
    fred = Fred(api_key=fred_key)
    raw_data = fred.get_series('CPIAUCSL').dropna().tail(12)

    df = pd.DataFrame(raw_data, columns=['Price_Index'])
    df['Month_Index'] = np.arange(len(df))

    # 4. AI ENGINE: Linear Regression Prediction
    X = df[['Month_Index']].values
    y = df['Price_Index'].values
    model = LinearRegression().fit(X, y)

    next_month = np.array([[len(y)]])
    prediction = model.predict(next_month)[0]

    # 5. DASHBOARD: Show the Stats
    col1, col2 = st.columns(2)
    col1.metric("Current Index", f"{y[-1]:.2f}")
    col2.metric("AI Forecast", f"{prediction:.2f}", f"{prediction - y[-1]:.2f}")

    # Visual Chart
    fig = px.line(df, x='Month_Index', y='Price_Index', title="Inflation Trend (Last 12 Months)")
    fig.add_scatter(x=[len(y)], y=[prediction], mode='markers', name='AI Prediction', marker=dict(size=12, color='red'))
    st.plotly_chart(fig)

    # 6. CHATBOT: The "Voice" of your Startup
    st.divider()
    st.subheader("💬 Ask the Econ-Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: Why is the AI predicting a price hike?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 7. AI RESPONSE LOGIC (FIXED URL & INDEX)
        # Note the /api/v1 at the end of the URL
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=ai_key)
        context = f"Live Data: Current CPI {y[-1]:.2f}, AI Forecast {prediction:.2f}."

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a Finance Expert for 8th graders. Use this context: {context}"},
                {"role": "user", "content": prompt}
            ]
        )

        # Note the [0] to pick the first AI message
        full_response = response.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.warning("Startup Engine Offline. Check API keys.")
