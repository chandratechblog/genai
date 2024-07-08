import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import GooglePalmEmbeddings

load_dotenv()

# Load the GROQ and Google API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Conversational PDF Document Q&A")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    You are an assistant for an insurance company "Generali"
    Please answer the questions based on the provided context only, please decline to answer if the question is not related to the context provided. If you are unsure about the answer, please ask for the information from the user and answer as appropriate. Please don't rename or change the product name even though the user asked for it. Please do not suggest any questions as response. 
    If they are asking about premium plans and related information please ask about the gender, age, and habits of the person. 
    Please provide the most accurate response based on the question and do not give any unethical, abusive, or illegal answers.
    <conversation_history>
    {conversation_history}
    </conversation_history>
    <context>
    {context}
    </context>
    Questions: {input}
    """
)

def vector_embedding(files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # st.session_state.embeddings = GooglePalmEmbeddings(model="models/embedding-gecko-001")
        
  
        # Load all PDFs and combine them into a single document set
        all_docs = []
        for file_path in files:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI embeddings

# File uploader for PDFs
uploaded_files = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and st.button("Save and Perform Embedding"):
    temp_files = []
    for uploaded_file in uploaded_files:
        temp_file_path = f"./temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        temp_files.append(temp_file_path)
    vector_embedding(temp_files)
    st.write("Vector Store DB is ready")

# Handling user question and response display
if "response_history" not in st.session_state:
    st.session_state.response_history = []

def get_response(question):
    conversation_history = ""
    for entry in st.session_state.response_history:
        conversation_history += f"User: {entry['question']}\nBot: {entry['response']}\n"
    
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({
        'input': question,
        'context': " ",  # Provide some initial context if needed
        'conversation_history': conversation_history
    })
    response_time = time.process_time() - start
    st.session_state.response_time = response_time
    return response
# <a target="_blank" href="https://icons8.com/icon/b3BL2Q2jPoHf/brave-ai">Brave AI</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>
# # Icons URLs
# user_icon_url = "https://raw.githubusercontent.com/iconic/open-iconic/master/svg/person.svg"
# bot_icon_url = "https://icons8.com/icon/b3BL2Q2jPoHf/brave-ai"

# Chat interface for user input and displaying responses
chat_container = st.container()
with chat_container:
    for i, entry in enumerate(st.session_state.response_history):
        st.chat_message("user").write(entry['question'])
        st.chat_message("assistant").write(entry['response'])
        with st.expander("Document Similarity Search", expanded=False):
            for doc in entry["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")
        feedback = st.empty()
        col1, col2 = feedback.columns([1, 0.1])
        if col1.button("👍", key=f"thumbs_up_{i}"):
            st.write("Thanks for your feedback!")
        elif col2.button("👎", key=f"thumbs_down_{i}"):
            st.write("Regenerating the answer...")
            response = get_response(entry['question'])
            entry['response'] = response['answer']
            entry['context'] = response['context']
            st.rerun()

if "vectors" in st.session_state:
    prompt1 = st.chat_input("Enter your question")
    if prompt1:
        response = get_response(prompt1)
        st.session_state.response_history.append({"question": prompt1, "response": response['answer'], "context": response["context"]})
        st.rerun()
