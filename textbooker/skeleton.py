"""
Skeleton for a Streamlit app that can do basic chat.
"""

from time import sleep
import streamlit as st
from important.llmsdk.openai import ChatOpenAI


with st.sidebar:
    model_name = st.selectbox(
        "Model",
        ["gpt-3.5-turbo-0613",
         "gpt-3.5-turbo-0301",
         "gpt-3.5-turbo-16k",
         "gpt-4-1106-preview",
         "gpt-4-32k-0314",]
        )
    
    if "chat" not in st.session_state:
        from dotenv import load_dotenv
        load_dotenv("../.env")
        st.session_state["chat"] = ChatOpenAI(model_name=model_name)
        
    else:
        chat = st.session_state["chat"]
        chat.model_name = model_name

st.title("Text Booker")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask some question"):
    st.session_state.messages.append(dict(content=prompt, role="user"))
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        messages=[{"role": m["role"], "content": m["content"]}
                      for m in st.session_state.messages]
        full_response = ""
        for token in chat.stream(messages):
            full_response += token
            message_placeholder.markdown(full_response + "▌")
            
        if full_response[-1] == "▌":
            full_response = full_response[:-1]
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
