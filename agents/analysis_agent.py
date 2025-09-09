class AnalysisAgent:
    def __init__(self):
        pass

    def analyze(self, research_hits, query):
        if not research_hits:
            return {"summary": "No data to analyze.", "comparisons": [], "recommended": None, "confidence": 0.0}
        for r in research_hits:
            r["norm_score"] = max(0.0, min(1.0, r.get("score", 0.0)))
        comparisons = []
        for r in research_hits:
            text_len = len(r.get("text",""))
            comparisons.append({
                "id": r["id"],
            "title": r.get("title",""),
            "score": r["norm_score"],
            "text_length": text_len
            })
        comparisons.sort(key=lambda x: (x["score"], x["text_length"]), reverse=True)
        recommended = comparisons[0]
        top_k = comparisons[:3]
        parts = []
        for c in top_k:
            parts.append(f"{c['title']} (score={c['score']:.2f}, len={c['text_length']})")
        summary = "Top sources: " + "; ".join(parts) + "."
        summary += f" Recommendation: prefer '{recommended['title']}' for this query."
        avg_conf = sum(c["score"] for c in top_k) / len(top_k) if top_k else 0.0
        return {"summary": summary, "comparisons": comparisons, "recommended": recommended, "confidence": float(avg_conf)}
