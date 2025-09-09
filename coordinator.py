import time
from datetime import datetime

class CoordinatorV3:
    def __init__(self, research_agent, analysis_agent, memory_agent, out_dir):
        self.research = research_agent
        self.analysis = analysis_agent
        self.memory = memory_agent
        self.trace = []
        self.out_dir = out_dir

    def _log(self, msg):
        ts = int(time.time())
        self.trace.append({"ts": ts, "msg": msg})
        print(f"[{ts}] {msg}")

    def handle(self, query):
        self._log(f"Received: {query}")
        plan = ["research", "analysis"] if "compare" in query.lower() or "analyze" in query.lower() else ["research"]
        self._log(f"Plan: {plan}")

        research_hits = self.research.search(query, top_k=6)
        self._log(f"Research hits: {len(research_hits)}")

        analysis_res = None
        if "analysis" in plan:
            analysis_res = self.analysis.analyze(research_hits, query)
            self._log("Analysis done")

        final_lines = []
        if analysis_res:
            final_lines.append("Analysis summary:\n" + analysis_res["summary"])
        final_lines.append("\nTop sources with provenance:")
        for r in research_hits[:5]:
            final_lines.append(f"- [{r['source']}] {r['title'] or r['id']} (score={r['score']:.3f})")

        final_text = "\n".join(final_lines)
        confidence = analysis_res["confidence"] if analysis_res else (research_hits[0]["confidence"] if research_hits else 0.0)

        topic = (research_hits[0]["title"] if research_hits and research_hits[0].get("title") else query[:60])
        mem = self.memory.store_fact(
            topic=topic,
            text=final_text,
            source="CoordinatorV3",
            agent="CoordinatorV3",
            confidence=float(confidence)
        )
        self._log(f"Stored memory id: {mem['id']} (topic: {topic})")

        return {
            "final_text": final_text,
            "final_confidence": float(confidence),
            "memory_id": mem["id"],
            "trace": self.trace
        }
