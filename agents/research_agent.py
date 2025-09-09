import json
from pathlib import Path

class ResearchAgent:
    def __init__(self, vector_store, kb_path):
        self.vs = vector_store
        self.kb_path = Path(kb_path)
        with open(self.kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        self.title_map = {d["id"]: d.get("title","") for d in kb}

    def search(self, query, top_k=5):
        vec_hits = self.vs.search_vector(query, top_k=top_k)
        results = []
        for h in vec_hits:
            doc_id = h["id"]
            results.append({
                "id": doc_id,
                "title": self.title_map.get(doc_id, ""),
                "text": h["text"],
                "score": float(h["score"]),
                "source": "kb" if str(doc_id).startswith("doc") else "memory/web",
                "agent": "ResearchAgent",
                "confidence": float(max(0.0, min(1.0, h["score"])))
            })
        return results
