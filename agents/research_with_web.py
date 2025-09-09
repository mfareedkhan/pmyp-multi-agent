import json
from pathlib import Path

class ResearchAgentWithWeb:
    def __init__(self, vector_store, kb_path, mock_web_path):
        self.vs = vector_store
        self.kb_path = Path(kb_path)
        self.mock_path = Path(mock_web_path)
        with open(self.kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        self.title_map = {d["id"]: d.get("title","") for d in kb}
        with open(self.mock_path, "r", encoding="utf-8") as f:
            self.mock_docs = json.load(f)
        for md in self.mock_docs:
            if md["id"] not in self.vs.ids:
                self.vs.add(md["id"], md["text"])
                self.title_map[md["id"]] = md.get("title","")

    def search(self, query, top_k=6):
        vec_hits = self.vs.search_vector(query, top_k=top_k)
        results = []
        for h in vec_hits:
            doc_id = h["id"]
            if str(doc_id).startswith("doc"):
                src = "kb"
            elif str(doc_id).startswith("web"):
                src = "web"
            else:
                src = "memory"
            results.append({
                "id": doc_id,
                "title": self.title_map.get(doc_id, ""),
                "text": h["text"],
                "score": float(h["score"]),
                "source": src,
                "agent": "ResearchAgent",
                "confidence": float(max(0.0, min(1.0, h["score"])))
            })
        return results
