import os
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
import numpy as np
import ollama
import fitz  # For PDF handling
from PIL import Image
import base64
import requests

class RAGChatbot:
    def __init__(self, ollama_model='mistral'):
        self.doc_texts = []
        self.doc_embeds = None
        self.chunks = []
        self.images = []
        self.ollama_model = ollama_model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text

    def image_to_base64(self, path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def load_files(self, file_paths):
        self.doc_texts = []
        self.images = []
        for fp in file_paths:
            if fp.lower().endswith(".pdf"):
                text = self.extract_text_from_pdf(fp)
                self.doc_texts.append(text)
            elif fp.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_b64 = self.image_to_base64(fp)
                self.images.append(img_b64)
            else:
                with open(fp, "r", encoding="utf-8") as f:
                    text = f.read()
                    self.doc_texts.append(text)
        self.chunks = []
        for doc in self.doc_texts:
            self.chunks.extend([chunk for chunk in doc.split("\n\n") if chunk.strip()])
        if self.chunks:
            self.doc_embeds = self.embedder.encode(self.chunks)
        else:
            self.doc_embeds = None

    def retrieve(self, query, k=2):  # use k=2 for faster responses
        if self.doc_embeds is None or not self.chunks:
            return []
        q_embed = self.embedder.encode([query])[0]
        sims = np.dot(self.doc_embeds, q_embed) / (
            np.linalg.norm(self.doc_embeds, axis=1) * np.linalg.norm(q_embed) + 1e-8
        )
        top_k_idx = sims.argsort()[-k:][::-1]
        return [self.chunks[i] for i in top_k_idx]

    def ask(self, question):
        if (self.doc_embeds is None or not self.chunks) and not self.images:
            return "Please upload and process a document or image first, or try web search."
        context = "\n".join(self.retrieve(question, k=2)) if self.chunks else ""
        if self.images:
            img_prompt = f"Given these images, answer: {question}. If context from documents exists, use it: {context}"
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": img_prompt}],
                images=self.images
            )
        else:
            prompt = (
                f"Use the following context to answer the question as thoroughly as possible. "
                f"\nContext:\n{context}\nQuestion: {question}\nDetailed answer:"
            )
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{"role": "user", "content": prompt}]
            )
        return response['message']['content'].strip()

    def ask_with_web(self, question):
        # DuckDuckGo API (no key, free, instant answer)
        url = f"https://api.duckduckgo.com/?q={question}&format=json"
        try:
            resp = requests.get(url, timeout=10)
            data = resp.json()
            abstract = data.get("AbstractText", "")
            related = data.get("RelatedTopics", [])
            snippet = abstract or (related[0]["Text"] if related and "Text" in related[0] else "")
        except Exception:
            snippet = ""
        context = f"Latest web information:\n{snippet if snippet else '[Could not find web answer]'}"
        prompt = (
            f"Use the following web context to answer the question as thoroughly as possible."
            f"\nWeb context:\n{context}\nQuestion: {question}\nAnswer:"
        )
        response = ollama.chat(
            model=self.ollama_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()
