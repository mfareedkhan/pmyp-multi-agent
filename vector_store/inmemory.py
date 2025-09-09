from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
import json

class InMemoryVectorStore:
    def __init__(self, model_name_or_obj='all-MiniLM-L6-v2'):
        if isinstance(model_name_or_obj, str):
            self.model = SentenceTransformer(model_name_or_obj)
        else:
            self.model = model_name_or_obj
        self.ids = []
        self.texts = []
        self.vectors = None

    def load_kb(self, kb_path: str, emb_path: str):
        kb_path = Path(kb_path)
        emb_path = Path(emb_path)
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        embs = np.load(emb_path)
        if len(kb) != len(embs):
            raise ValueError("KB length and embeddings length mismatch")
        self.ids = [d["id"] for d in kb]
        self.texts = [d["text"] for d in kb]
        self.vectors = np.array(embs)

    def add(self, doc_id: str, text: str):
        vec = self.model.encode([text], show_progress_bar=False)[0]
        self.ids.append(doc_id)
        self.texts.append(text)
        if self.vectors is None:
            self.vectors = np.array([vec])
        else:
            self.vectors = np.vstack([self.vectors, vec])
        return vec

    def search_vector(self, query: str, top_k: int = 3):
        if self.vectors is None or len(self.ids) == 0:
            return []
        qvec = self.model.encode([query], show_progress_bar=False)[0]
        sims = cosine_similarity([qvec], self.vectors)[0]
        idxs = sims.argsort()[::-1][:top_k]
        results = []
        for i in idxs:
            results.append({"id": self.ids[i], "text": self.texts[i], "score": float(sims[i])})
        return results

    def search_keyword(self, keyword: str):
        keyword = keyword.lower()
        results = []
        for i, txt in enumerate(self.texts):
            if keyword in txt.lower():
                results.append({"id": self.ids[i], "text": txt})
        return results
