import streamlit as st
import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_google_vertexai import ChatVertexAI

from vertexai.language_models import ChatModel
from langchain_core.messages import AIMessage,HumanMessage


model_id='chat-bison@002'
context='''
You are a helpful order return assistant.
Customers will use you to return their items.Items can only be returned if they were purchased within the lats 7 days and are unsued.
Make sure to confirm that item is both unused and was purchased within the lats 7 days. Please ask your customer for both rules.
If both the above conditions are met, then show a message with the return address(1st main, bangalore, 562114).Otherwise reject the rertun
with a freindly formatted message. Do not worry about return order numbers or product details.
'''
chat_model = ChatModel.from_pretrained(model_id)
chat = chat_model.start_chat(context=context)
parameters={
    'temperature': 0.0,
    'max_output_tokens': 256,
    'top_p':0.9,
    'top_k': 40
}

def get_response(prompt, parameters, messages):
    history=''
    for message in messages:
        if isinstance(message, HumanMessage):
            history=f'{history}\nUser:{message.content}'

        if isinstance(message, AIMessage):
            history=f'{history}\AAssistant:{message.content}'

    response =  chat.send_message(f'{history}\n{prompt}', **parameters)
    response_text = response.text

    for word in response_text.split():
        yield word + ' '
        time.sleep(0.1)

def reset_chat():
    st.session_state.messages = []

def main():
    st.title("Customer Service Agent Demo")
    st.write("Hello, Welcome to the return center. I want to help you.")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message('Human'):
            st.markdown(message.content)


    if prompt := st.chat_input('Hello, how can help you?'):
        st.session_state.messages.append(HumanMessage(prompt))
        with st.chat_message('Human'):
            st.markdown(prompt)

        with st.chat_message('AI'):
            response = get_response(prompt, parameters, st.session_state.messages)
        st.session_state.messages.append(AIMessage(response)) 

    st.button('Reset chat', on_click=reset_chat)

if __name__ == '__main__':
    main()