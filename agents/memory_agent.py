import time, uuid, json
from pathlib import Path

def now_ts():
    return int(time.time())

class MemoryAgent:
    def __init__(self, vector_store, memory_file: str):
        self.vs = vector_store
        self.memory_file = Path(memory_file)
        if self.memory_file.exists():
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        else:
            self.records = []
            self._save()

    def _save(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def store_fact(self, topic: str, text: str, source: str, agent: str, confidence: float):
        rec = {
            "id": str(uuid.uuid4()),
            "timestamp": now_ts(),
            "topic": topic,
            "text": text,
            "source": source,
            "agent": agent,
            "confidence": float(confidence)
        }
        self.vs.add(rec["id"], rec["text"])
        self.records.append(rec)
        self._save()
        return rec

    def search_by_topic(self, topic_keyword: str):
        keyword_hits = [r for r in self.records if topic_keyword.lower() in (r.get("topic","").lower() + " " + r.get("text","").lower())]
        vs_hits = self.vs.search_vector(topic_keyword, top_k=3)
        return {"keyword_hits": keyword_hits, "vector_hits": vs_hits}

    def similarity_check(self, query: str, threshold: float = 0.75):
        hits = self.vs.search_vector(query, top_k=5)
        filtered = [h for h in hits if h["score"] >= threshold]
        return filtered
