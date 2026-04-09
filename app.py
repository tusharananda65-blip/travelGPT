import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from backend import get_answer, get_vector_store
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Travel Guide AI",
    layout="wide"
)

# UI Styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🌍 Travel Guide AI ✈️")
st.caption("Ask anything about destinations, itineraries, budgets, and travel tips.")

st.divider()

# Sidebar
st.sidebar.header("Controls")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Warm up vector DB
@st.cache_resource(show_spinner=False)
def warm_up():
    return get_vector_store()

with st.spinner("⏳ Loading travel knowledge base..."):
    vector_store = warm_up()

# Travel query classifier
def detect_query_type(query):
    q = query.lower()

    if "itinerary" in q or "plan" in q:
        return "Travel Itinerary"
    elif "budget" in q or "cost" in q:
        return "Budget Planning"
    elif "hotel" in q or "stay" in q:
        return "Accommodation"
    elif "food" in q or "restaurant" in q:
        return "Food & Dining"
    elif "places" in q or "visit" in q:
        return "Tourist Attractions"
    elif "transport" in q or "how to reach" in q:
        return "Transport & Travel"
    else:
        return "General Travel Query"

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
user_input = st.chat_input("Ask your travel question (e.g., '3-day Goa itinerary')...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("🌍 Planning your trip..."):

            # Detect type
            query_type = detect_query_type(user_input)

            # Retrieve docs
            retrieved_docs = vector_store.similarity_search(user_input, k=3)

            # Show retrieved context
            with st.sidebar:
                st.subheader("📄 Retrieved Travel Context")
                if retrieved_docs:
                    for i, doc in enumerate(retrieved_docs, start=1):
                        st.markdown(f"**Doc {i}:**")
                        st.write(doc.page_content[:400] + "...")
                else:
                    st.write("No relevant documents found.")

            # Get answer
            answer = get_answer(user_input)

            # Output
            st.markdown(f"**🧭 Query Type:** {query_type}")
            st.markdown("### ✨ Travel Advice")
            st.markdown(answer)

            # Copy button
            copy_script = f"""
            <button onclick="navigator.clipboard.writeText(`{answer}`)">
            📋 Copy Answer
            </button>
            """
            components.html(copy_script, height=40)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()

# Footer
st.caption(
    "⚠️ This tool provides general travel suggestions. Prices, availability, and conditions may vary. "
    "Always verify before booking."
)
