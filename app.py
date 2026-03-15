import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from openai import OpenAI, AuthenticationError
import plotly.express as px

# 1. SECURE CREDENTIALS
try:
    fred_key = st.secrets["FRED_KEY"]
    ai_key = st.secrets["AI_KEY"]
except:
    st.error("🔑 Keys not found! Go to Settings → Secrets and add FRED_KEY and AI_KEY.")
    st.stop()

# 2. DEBUG: Verify key format (remove after confirming it works)
if not ai_key.startswith("sk-or-"):
    st.warning(f"⚠️ AI_KEY looks wrong. It starts with `{ai_key[:8]}...` — OpenRouter keys start with `sk-or-v1-`. Check your Streamlit secrets.")

# 3. BRANDING & HEADER
st.title("📈 TCPI: Consumer Price Index")
st.caption("🚀 TCPI Startup | Status: Live & Automated")
st.markdown("### *AI-Powered Economic Forecasting for the Next Generation*")

# 4. DATA ENGINE: Fetch Live Inflation Data
try:
    fred = Fred(api_key=fred_key)
    raw_data = fred.get_series('CPIAUCSL').dropna().tail(12)
except Exception as e:
    st.error(f"📡 FRED data fetch failed. Check your FRED_KEY. Error: {e}")
    st.stop()

df = pd.DataFrame(raw_data, columns=['Price_Index'])
df['Month_Index'] = np.arange(len(df))

# 5. AI ENGINE: Linear Regression Prediction
X = df[['Month_Index']].values
y = df['Price_Index'].values
model = LinearRegression().fit(X, y)
next_month = np.array([[len(y)]])
prediction = model.predict(next_month)[0]

# 6. DASHBOARD: Show the Stats
col1, col2 = st.columns(2)
col1.metric("Current Index", f"{y[-1]:.2f}")
col2.metric("AI Forecast", f"{prediction:.2f}", f"{prediction - y[-1]:+.2f}")

# Visual Chart
fig = px.line(df, x='Month_Index', y='Price_Index', title="Inflation Trend (Last 12 Months)")
fig.add_scatter(
    x=[len(y)], y=[prediction],
    mode='markers', name='AI Prediction',
    marker=dict(size=12, color='red')
)
st.plotly_chart(fig)

# 7. CHATBOT
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

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=ai_key
        )

        context = f"Live Data: Current CPI {y[-1]:.2f}, AI Forecast {prediction:.2f}."
        system_msg = {
            "role": "system",
            "content": f"You are a friendly Finance Expert explaining economics to 8th graders. Keep answers short and clear. Use this live data context: {context}"
        }

        # Pass full conversation history for memory
        all_messages = [system_msg] + st.session_state.messages

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=all_messages
        )

        full_response = response.choices[0].message.content

    except AuthenticationError:
        full_response = (
            "❌ **Authentication failed.** Your OpenRouter API key was rejected.\n\n"
            "**Fix it:**\n"
            "1. Go to [openrouter.ai/keys](https://openrouter.ai/keys) and copy your key\n"
            "2. In Streamlit → Settings → Secrets, update `AI_KEY`\n"
            "3. Keys must start with `sk-or-v1-...`\n"
            "4. Make sure your account has credits"
        )
    except Exception as e:
        full_response = f"⚠️ Something went wrong: {e}"

    with st.chat_message("assistant"):
        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
