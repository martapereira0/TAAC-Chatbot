from dotenv import load_dotenv
import os
import shutil
import tempfile
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from typing import Annotated
import streamlit as st

# Carregar variáveis de ambiente
load_dotenv()
os.environ["USER_AGENT"] = "RAG-Chatbot"
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# 2. Configuração de Modelos e Pipeline
llm = ChatMistralAI(model="open-mistral-7b", api_key=mistral_api_key)

# Verifica se o histórico existe no estado da sessão, se não, inicializa
if 'history' not in st.session_state:
    st.session_state.history = []

# Maximum number of responses to store in history
max_history_length = 10

def format_docs(docs):
    """Formata documentos carregados numa string para entrada no modelo."""
    return "\n\n".join(doc.page_content for doc in docs)

def load_and_process_document(uploaded_file):
    """Carrega e processa o documento PDF em chunks para uso no modelo."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        shutil.copyfileobj(uploaded_file, tmp_file)
        tmp_file_path = tmp_file.name

    
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    return vectorstore

def retrieve_and_generate_response(query, vectorstore):
    """Recupera documentos relevantes e gera resposta com histórico."""
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.0}
    )

    retrieved_docs = retriever.get_relevant_documents(query)
    context = format_docs(retrieved_docs) if retrieved_docs else "No relevant documents found."
    
    # Preparar histórico
    if len(st.session_state.history) > 1:
        # Se houver mais de uma resposta, preenche previous_responses com todas as respostas, exceto a última
        previous_responses = st.session_state.history[:(len(st.session_state.history)-1)]
        previous_text = "\n".join(f"Previous Answer {i+1}: {ans}" for i, ans in enumerate(previous_responses))
        last_response = st.session_state.history[-1]  # A última resposta recente
    else: 
        previous_responses = previous_text = last_response = st.session_state.history[-1] if st.session_state.history else "No previous answer available."
        
    print("previous_responses:", previous_responses)
    print("previous_text:", previous_text)
    print("last_response:", last_response)

    # Formatar prompt
    prompt_template = PromptTemplate(
        input_variables=["previous", "last", "context", "question"],
        template="Previous Answers:\n{previous}\n\nLast Answer:\n{last}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt = prompt_template.format(previous=previous_text, last=last_response, context=context, question=query)
    print("PROMPT:", prompt, end="\n\n")
    # Gerar resposta e atualizar histórico
    response = llm.invoke(prompt)

    
    st.session_state.history.append(response.content)
    print("st.session_state.history:", st.session_state.history, end="\n\n\n\n")

    if len(st.session_state.history) > max_history_length:
        st.session_state.history.pop(0)
    
    return response.content

st.title("Ask your Data")

if 'messages' not in st.session_state:
    st.session_state.messages = []


uploaded_file = st.file_uploader("Upload your PDF document", type=["txt", "pdf", "docx"])

user_prompt = st.chat_input("Pass your prompt here.")

if uploaded_file:
    st.write("File uploaded successfully!")
    
    vector_store = load_and_process_document(uploaded_file)

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    if user_prompt:
        st.chat_message('user').markdown(user_prompt)
        st.session_state.messages.append({'role': 'user', 'content': user_prompt})

        # Gerar resposta usando a cadeia RAG e histórico
        response = retrieve_and_generate_response(user_prompt, vector_store)
        
        if response:
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            

else:
    st.write("Please upload a file to continue.")
