Multi-PDF RAG Chatbot 🤖📚

A powerful Retrieval-Augmented Generation (RAG) chatbot that processes multiple PDF documents simultaneously, extracts knowledge, and provides intelligent, context-aware responses using advanced AI models.

🌐 Live Demo
Live Application: https://multi-pdf-rag-chatbott.streamlit.app/

✨ Features
📄 Multi-PDF Processing
Upload and process multiple PDF files simultaneously

Support for various document formats (with PDF focus)

Automatic text extraction and chunking

Smart document splitting with overlap for context preservation

🧠 Advanced RAG Architecture
Hybrid Search: Combines semantic search with keyword matching

Context-Aware Retrieval: Maintains conversation context across queries

Relevance Scoring: Filters retrieved chunks by relevance threshold

Source Citation: Every response includes source document references

🤖 AI-Powered Intelligence
OpenAI GPT-3.5/GPT-4 integration

Customizable model parameters (temperature, max tokens)

Streaming responses for better user experience

Conversation memory with adjustable history length

🎨 User-Friendly Interface
Clean, intuitive Streamlit web interface

Real-time file processing progress indicators

Interactive chat interface with message history

Mobile-responsive design

Dark/Light mode support

🔧 Advanced Capabilities
Document Summarization: Quick summaries of uploaded documents

Question Answering: Context-aware Q&A based on document content

Cross-Document Analysis: Connect information across multiple PDFs

Export Conversations: Save chat history for future reference
<img width="633" height="517" alt="image" src="https://github.com/user-attachments/assets/1e2470b4-2dc6-4e92-9a42-1ded49b7ed08" />

TECH-STACK
<img width="698" height="441" alt="image" src="https://github.com/user-attachments/assets/f12e5214-c501-472a-b6e2-bb4e1bfb07a2" />

📦 Installation
Prerequisites
Python 3.8 or higher

OpenAI API key

Git

Local Setup
Clone the repository

bash
git clone https://github.com/Nikitha2341/Multi-pdf-RAG-Chatbot.git
cd Multi-pdf-RAG-Chatbot
Create virtual environment

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Set up environment variables
Create a .env file in the root directory:

env
OPENAI_API_KEY=sk-your-openai-api-key-here
Run the application

bash
streamlit run app.py
🚀 Deployment
Streamlit Cloud (Recommended)
Push to GitHub

bash
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
Deploy on Streamlit Cloud

Go to share.streamlit.io

Click "New app"

Select your repository and branch

Set main file to app.py

Add secrets in Settings → Secrets:

toml
OPENAI_API_KEY = "your-actual-openai-key"
Access your app




