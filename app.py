import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

load_dotenv()

# --- 1. ROBUST API KEY FINDER ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = os.getenv("gemini_api_key")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)

# --- 2. SMART MODEL FINDER (Updated to avoid experimental models) ---
def get_available_model_options():
    """Returns a list of available models and the best default one."""
    default_model = "models/gemini-1.5-flash"
    fallback_options = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "gemini-1.5-flash"]
    
    try:
        if not api_key: 
            return fallback_options, default_model
        
        # Get all models that support generating content
        all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Filter out experimental/preview models that often have 0 quota
        stable_models = [m for m in all_models if "exp" not in m and "preview" not in m]
        
        # If we filtered everything out, just use whatever we found
        if not stable_models:
            stable_models = all_models
            
        if not stable_models:
            return fallback_options, default_model

        # Determine best default
        best_model = stable_models[0]
        for m in stable_models:
            if "1.5" in m and "flash" in m: # Prefer 1.5 Flash
                best_model = m
                break
        
        return stable_models, best_model
            
    except Exception as e:
        print(f"Error listing models: {e}")
        return fallback_options, default_model


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        embeddings.embed_query("test")
    except:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    vectorstore = None
    batch_size = 100
    
    progress_text = "Creating vector store..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, start_index in enumerate(range(0, len(text_chunks), batch_size)):
        batch = text_chunks[start_index : start_index + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_texts(texts=batch, embedding=embeddings)
        else:
            vectorstore.add_texts(batch)
        time.sleep(1)

    vectorstore.save_local("faiss_index")
    my_bar.empty()

def get_conversational_chain(model_name):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not found in the context, politely say "I am sorry, I could not find any information on that topic from the provided document\n\n
    Context: \n {context}?\n
    Question: \n {question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=0.3, 
        google_api_key=api_key
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, model_name):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    except:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(model_name)
        
        # Retry logic for rate limits
        response = None
        for attempt in range(3):
            try:
                response = chain(
                    {"input_documents":docs, "question": user_question},
                    return_only_outputs=True
                )
                break
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    raise e

        if response:
            st.write("Reply: ", response["output_text"])
        else:
            st.error("Server is busy. Please try again.")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    st.set_page_config("Chat with Multiple PDF", page_icon=":books:")
    st.header("Chat with Multiple PDF using Gemini")

    with st.sidebar:
        st.title("Settings")
        if api_key:
            st.success("API Key Found")
        else:
            st.error("API Key Missing")

        # --- MODEL SELECTOR (Restored) ---
        # Get valid models and the best default
        model_options, default_model = get_available_model_options()
        
        # Ensure default is in options
        if default_model not in model_options:
            model_options.insert(0, default_model)
            
        # Let user override if needed
        selected_model = st.selectbox(
            "Select Chat Model:", 
            options=model_options, 
            index=model_options.index(default_model)
        )
        
        st.info(f"Active Model: {selected_model}")

        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not api_key:
                st.error("Please set your API Key first.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vectorstore(text_chunks)
                    st.success("Done")

    user_question = st.text_input("Ask your question about the document")

    if user_question:
        if not api_key:
            st.error("API Key is missing.")
        else:
            user_input(user_question, selected_model)

if __name__ == "__main__":
    main()