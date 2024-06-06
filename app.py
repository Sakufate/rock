import streamlit as st
#python-dotenv
import sys
from dotenv import load_dotenv
#import dotenv
import getpass
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
#from langchain.chains.history_aware_retriever import  create_history_aware_retriever
#from langchain.chains.retrieval import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
#from langchain_chroma import Chroma
#from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from htmlTemplates import css, bot_template, user_template
#langchain_core
#streamlit run app.py
#if not os.environ.get("OPENAI_API_KEY"):
    #os.environ["OPENAI_API_KEY"] = getpass.getpass()


#llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


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
    #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chan = ConversationalRetrievalChain.from_llm(llm = llm, retriever = vectorstore.as_retriever(), memory = memory)   
    return conversation_chan

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if 1 % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    #OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    #OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
    #print(os.getenv("OPENAI_API_KEY"))
    #os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    #if not os.environ.get("OPENAI_API_KEY"):
    #    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
    
    

    st.set_page_config(page_title="Scan knowledge base", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Scan knwoledge base :books:")
    user_question = st.text_input("Input the question for the knowledge base")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Knowledge base documents")
        pdf_docs = st.file_uploader("Upload files", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()