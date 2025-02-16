import streamlit as st
import os
import json
import torch
import shutil
import ffmpeg
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import whisper

# ✅ Qdrant API Configuration
QDRANT_CLOUD_URL = "https://a03a15ac-6de0-4fcf-b206-8d0354c2827b.us-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3NDcwNDI4fQ.gwrFSabmd34SQwSzccDex4A0wkFPahaGL8NfgTezE-Y"

# ✅ Initialize Qdrant
qdrant = QdrantClient(QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "video_rag"

# ✅ Ensure Collection Exists
collections = qdrant.get_collections()
if COLLECTION_NAME not in [col.name for col in collections.collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
else:
    print(f"⚠️ Collection `{COLLECTION_NAME}` already exists. Skipping creation.")

# ✅ Load Models
device = 0 if torch.cuda.is_available() else -1
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small", device=device)
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
whisper_model = whisper.load_model("base", download_root="models", local_files_only=True)

# ✅ Extract Frames Using PIL (Replaces OpenCV)
def extract_frames(video_path, output_folder, fps=0.2):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    vidcap = ffmpeg.input(video_path)
    frame_count = 0
    frame_rate = int(30 / fps)

    for frame in vidcap.output("pipe:", format="image2pipe", vframes=100):
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        with open(frame_path, "wb") as f:
            f.write(frame)
        frame_count += 1
        if frame_count >= 100:  # Stop after 100 frames for efficiency
            break

    print("✅ Frames extracted successfully!")

# ✅ Extract Audio Using FFmpeg-Python
def extract_audio(video_path, audio_path):
    ffmpeg.input(video_path).output(audio_path, format="mp3", acodec="libmp3lame").run(overwrite_output=True)
    print("✅ Audio extracted successfully!")

# ✅ Transcribe Audio
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# ✅ Generate Captions Efficiently
def generate_captions(frames_folder):
    captions = {}
    batch_images = []

    for frame in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame)
        if frame.endswith((".jpg", ".png")):
            image = Image.open(frame_path).convert("RGB")
            batch_images.append((frame, image))

    inputs = caption_processor(batch_images, return_tensors="pt", padding=True)
    captions_output = caption_model.generate(**inputs)

    for i, (frame, _) in enumerate(batch_images):
        captions[frame] = caption_processor.decode(captions_output[i], skip_special_tokens=True)

    return captions

# ✅ Store Embeddings in Qdrant
def store_embeddings(captions):
    points = []
    for idx, (frame, text) in enumerate(captions.items()):
        embedding = embed_model.encode(text).tolist()
        points.append({"id": idx, "vector": embedding, "payload": {"frame": frame, "text": text}})
    
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Embeddings stored successfully!")

# ✅ Search & Answer Questions
def search_video(query):
    query_embedding = embed_model.encode(query).tolist()
    search_results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=query_embedding, limit=1)

    if search_results:
        best_match = search_results[0]
        retrieved_frame = best_match.payload["frame"]
        retrieved_text = best_match.payload["text"]
        return retrieved_frame, retrieved_text
    return None, "No relevant content found."

def generate_answer(query, context):
    prompt = f"Based on this video segment:\n{context}\n\nAnswer the question: {query}"
    response = qa_pipeline(prompt, max_length=100)[0]["generated_text"]
    return response

# ✅ Streamlit UI
st.title("📽️ VideoRAG - AI-Powered Video Q&A")

uploaded_file = st.file_uploader("📂 Upload a video", type=["mp4", "mov"])
if uploaded_file:
    video_path = f"data/{uploaded_file.name}"
    frames_folder = "data/frames_output"
    audio_path = "data/video_audio.mp3"

    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)
    st.write("🔄 Processing Video... (This may take a few minutes)")

    extract_frames(video_path, frames_folder)
    extract_audio(video_path, audio_path)

    st.write("🔊 **Extracting Audio & Transcribing...**")
    transcription = transcribe_audio(audio_path)
    st.write("✅ **Transcription:**", transcription)

    st.write("🖼️ **Generating Captions...**")
    captions = generate_captions(frames_folder)

    # Save Captions to File
    with open("data/captions.json", "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=4)

    st.write("📌 **Storing Embeddings in Qdrant...**")
    store_embeddings(captions)
    st.write("✅ **Embeddings Stored!**")

st.subheader("🤔 Ask a question about the video")
query = st.text_input("🔍 Type your question:")
if query:
    st.write("🔹 Searching the video for relevant information...")
    retrieved_frame, retrieved_text = search_video(query)

    if retrieved_text:
        answer = generate_answer(query, retrieved_text)
        st.subheader("💡 AI Answer:")
        st.write(answer)

        # Show the frame
        frame_path = os.path.join("data/frames_output", retrieved_frame)
        if os.path.exists(frame_path):
            st.image(frame_path, caption="Related Frame", use_column_width=True)
    else:
        st.error("❌ No relevant content found in the video.")
