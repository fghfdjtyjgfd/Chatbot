import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from rag import rag_chain

load_dotenv()

# app config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Chatbot")

def get_response(user_query, chat_history):
    return rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_query,
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="สวัสดี! ฉันเป็นแพทย์ AI ที่สามารถช่วยตอบคำถามเกี่ยวกับโรคและอาการของผู้ป่วยได้ กรุณาถามมาได้เลย!"),
    ]

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.chat_history)
        st.write(response["answer"])
    st.session_state.chat_history.append(AIMessage(content=response["answer"]))
