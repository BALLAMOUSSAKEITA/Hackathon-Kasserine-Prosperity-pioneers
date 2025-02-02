from flask import Flask, request, jsonify, render_template,send_from_directory
from flask_cors import CORS
import ollama
import fitz  # PyMuPDF pour extraire le texte des PDF
import faiss
import os
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from gtts import gTTS

app = Flask(__name__)
CORS(app)

AUDIO_FOLDER = "static/audio"
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

LEVEL_TEXTS = {
    1: "Bienvenue au Temple Romain de Sufetula. Ce monument majestueux était le centre religieux de la cité antique...",
    2: "Découvrez l'Arc de Triomphe de Sbeitla, un témoignage remarquable de l'architecture romaine...", 
    3: "Le Forum était le cœur battant de la ville romaine, où se déroulaient les activités politiques et commerciales...",
    4: "Explorez le Théâtre Romain antique, lieu de spectacles et de rassemblements culturels...",
    5: "Les Thermes Romains étaient essentiels à la vie sociale, offrant bains publics et espaces de détente..."
}

@app.route('/audio/<int:level>', methods=['GET', 'POST'])
def get_audio(level):
    audio_file = f"level_{level}.mp3"
    audio_path = os.path.join(AUDIO_FOLDER, audio_file)
    if request.method == 'POST':
        if level in LEVEL_TEXTS:
            tts = gTTS(text=LEVEL_TEXTS[level], lang='fr')
            tts.save(audio_path)
            app.logger.info(f"Generated audio file: {audio_path}")
    else:
        if not os.path.exists(audio_path):
            app.logger.info(f"Audio file not found: {audio_path}")
        else:
            app.logger.info(f"Serving existing audio file: {audio_path}")
    return send_from_directory(AUDIO_FOLDER, audio_file)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/quiz")
def quiz():
    return render_template("jeux.html", total_levels=len(LEVEL_TEXTS))

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def load_pdf_documents():
    pdf_texts = []
    for filename in os.listdir("docs"):
        if filename.endswith(".pdf"):
            pdf_texts.append(extract_text_from_pdf(f"docs/{filename}"))
    return pdf_texts

def split_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text("\n".join(texts))

if os.path.exists("vectorstore"):
    db = FAISS.load_local("vectorstore", OllamaEmbeddings(model="mistral"), allow_dangerous_deserialization=True)
else:
    texts = split_texts(load_pdf_documents())
    embeddings = OllamaEmbeddings(model="mistral")
    db = FAISS.from_texts(texts, embeddings)
    db.save_local("vectorstore")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features.numpy().flatten()

def load_and_index_images():
    image_paths = [os.path.join("images", f) for f in os.listdir("images") if f.endswith((".png", ".jpg", ".jpeg"))]
    if not image_paths:
        return None

    image_embeddings = [extract_image_features(img) for img in image_paths]
    image_embeddings = np.array(image_embeddings, dtype=np.float32)

    d = image_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(image_embeddings)
    faiss.write_index(index, "image_vectorstore.index")
    return index

if os.path.exists("image_vectorstore.index"):
    image_index = faiss.read_index("image_vectorstore.index")
else:
    image_index = load_and_index_images()

def text_to_speech(text, filename="static/response.mp3"):
    tts = gTTS(text=text, lang="fr")
    tts.save(filename)

@app.route('/send_message', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        image_query = data.get("image_query", False)

        if not user_input:
            return jsonify({"error": "Message manquant"}), 400

        context = ""
        results = db.similarity_search(user_input, k=3)
        context += "\n".join([doc.page_content for doc in results])

        if image_query and image_index is not None:
            query_embedding = extract_image_features("query.jpg")
            query_embedding = np.array([query_embedding], dtype=np.float32)
            _, I = image_index.search(query_embedding, k=1)

            if len(I) > 0 and len(I[0]) > 0:
                matched_image_idx = I[0][0]
                matched_image_path = os.listdir("images")[matched_image_idx]
                context += f"\n[Image trouvée: {matched_image_path}]"

        full_prompt = f"Contexte:\n{context}\n\nQuestion: {user_input}\nRéponse :"
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": full_prompt}])
        
        chatbot_response = response["message"]["content"]
        text_to_speech(chatbot_response)  

        return jsonify({"response": chatbot_response, "audio_url": "/static/response.mp3"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
