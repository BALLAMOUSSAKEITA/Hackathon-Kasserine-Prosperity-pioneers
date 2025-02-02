import fitz  # PyMuPDF pour extraire le texte
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# ⚠️ Mets ta clé OpenAI ici
os.environ["OPENAI_API_KEY"] = "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"

def extract_text_from_pdf(pdf_path):
    """Extrait le texte d'un PDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def create_vector_store(text_data):
    """Crée une base de données vectorielle FAISS."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts([text_data], embeddings)
    return vector_store

class RAGSystem:
    """Système de RAG pour rechercher et générer des réponses."""
    def __init__(self, pdf_path):
        text_data = extract_text_from_pdf(pdf_path)
        self.vector_store = create_vector_store(text_data)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            retriever=self.vector_store.as_retriever()
        )

    def get_response(self, user_query):
        return self.qa_chain.run(user_query)

def load_rag_system(pdf_path):
    """Charge et retourne le système RAG prêt à l'emploi."""
    return RAGSystem(pdf_path)
