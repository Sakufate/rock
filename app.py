import sys
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

#streamlit run app.py

def get_pdfs_from_resources():
    loader = PyPDFDirectoryLoader("resources/")
    docs = loader.load()
    return docs

def get_docs_text(docs):
    text = ""
    for file in docs:
        text += file.page_content
    return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
            separator="\n", 
            chunk_size = 1000, 
            chunk_overlap=200, 
            length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chan = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vectorstore.as_retriever(), memory = memory)   
    return conversation_chan

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
def update_button_clicked():
    st.session_state.button_clicked = True

def main():
    load_dotenv()
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 

    st.set_page_config(page_title="Scan knowledge base", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)


    if "data_not_ready" not in st.session_state:
        st.session_state.data_not_ready = True

    if "conversation" not in st.session_state:
        st.session_state.conversation = None    

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    with st.sidebar:
        st.subheader("Knowledge base documents")
        button = st.button("Process", on_click=update_button_clicked, disabled=st.session_state.button_clicked)

        if button:
            with st.spinner("Processing"):

                # get pdf docs from resources folder
                pdf_folder_docs = get_pdfs_from_resources()
                
                # get pdf docs from resources folder
                raw_text = get_docs_text(pdf_folder_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

                #allow to write the question for the knowledge base
                st.session_state.data_not_ready = not st.session_state.data_not_ready

                #show that processing is Done
                st.success('Processing Done!', icon="✅",)

    st.header("Scan knwoledge base :books:")

    user_question = st.text_input(label="Input the question for the knowledge base", disabled=st.session_state.data_not_ready)   

    if user_question:
        handle_userinput(user_question)    

if __name__ == '__main__':
    main()