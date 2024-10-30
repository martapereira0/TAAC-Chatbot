from dotenv import load_dotenv
import os
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate

import streamlit as st


load_dotenv()
os.environ["USER_AGENT"] = "RAG-Chatbot"
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# Load the document
pdf_file_path = "ArtificialIntelligenceAct.pdf"

loader = PyPDFLoader(pdf_file_path)

# Load and process the document
docs = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.8}
            )

prompt = hub.pull("rlm/rag-prompt")

llm = ChatMistralAI(model="open-mistral-7b", api_key=mistral_api_key)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Pipeline de RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Configuração da interface Streamlit
st.title("Ask our Chatbot")

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Build a prompt input template to display the prompts    
user_prompt = st.chat_input("Pass your prompt here")

# If the user hits the enter then
if user_prompt:
    st.chat_message('user').markdown(user_prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role': 'user', 'content': user_prompt})
    # Send the prompt to the LLM
    response = rag_chain.invoke(user_prompt)
    # Display the model's response
    st.chat_message('assistant').markdown(response)

    # Store the model's response in the state
    st.session_state.messages.append({'role': 'assistant', 'content': response})