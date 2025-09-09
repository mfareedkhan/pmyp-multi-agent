import json
from pathlib import Path
from vector_store.inmemory import InMemoryVectorStore
from agents.memory_agent import MemoryAgent
from agents.research_with_web import ResearchAgentWithWeb
from agents.analysis_agent import AnalysisAgent
from coordinator import CoordinatorV3
import numpy as np
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VSTORE = ROOT / "vector_store"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

KB_PATH = DATA / "kb.json"
EMB_PATH = VSTORE / "kb_embeddings.npy"
MOCK_PATH = DATA / "mock_web.json"
MEM_FILE = VSTORE / "memory.json"

def ensure_sample_kb():
    if not KB_PATH.exists():
        sample_kb = [
            {"id":"doc1","title":"Neural networks - overview","text":"Neural networks are a set of algorithms, modeled loosely after the human brain, designed to recognize patterns."},
            {"id":"doc2","title":"Convolutional Neural Networks","text":"Convolutional neural networks (CNNs) are specialized for processing grid-like data such as images."},
            {"id":"doc3","title":"Recurrent Neural Networks","text":"Recurrent neural networks (RNNs) are designed to work with sequential data and have internal memory."},
            {"id":"doc4","title":"Transformer architectures","text":"Transformers use attention mechanisms and are effective for many NLP tasks; they scale well and replace recurrence."},
            {"id":"doc5","title":"Optimization techniques","text":"Optimization techniques include gradient descent, stochastic gradient descent, Adam optimizer, and learning rate schedules."}
        ]
        KB_PATH.parent.mkdir(parents=True, exist_ok=True)
        KB_PATH.write_text(json.dumps(sample_kb, indent=2), encoding='utf-8')
    if not EMB_PATH.exists():
        # create embeddings using sentence-transformers
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = json.loads(KB_PATH.read_text(encoding='utf-8'))
        texts = [d['text'] for d in texts]
        embs = model.encode(texts, show_progress_bar=False)
        np.save(EMB_PATH, embs)

def main():
    ensure_sample_kb()
    vs = InMemoryVectorStore('all-MiniLM-L6-v2')
    vs.load_kb(str(KB_PATH), str(EMB_PATH))
    mem = MemoryAgent(vs, str(MEM_FILE))
    analysis = AnalysisAgent()
    research = ResearchAgentWithWeb(vs, str(KB_PATH), str(MOCK_PATH))
    coord = CoordinatorV3(research, analysis, mem, OUT)

    scenarios = {
        "simple_query": "What are the main types of neural networks?",
        "complex_query": "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs.",
        "memory_test": "What did we discuss about neural networks earlier?",
        "multi_step": "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.",
        "collaborative": "Compare two machine-learning approaches (CNN vs Transformer) and recommend which is better for image classification."
    }

    for name, q in scenarios.items():
        print("\nRunning:", name)
        out = coord.handle(q)
        header = f"Scenario: {name}\nQuery: {q}\nTime: {datetime.utcnow().isoformat()}Z\nPlan: {coord.plan if hasattr(coord,'plan') else 'auto'}\nMemory ID: {out['memory_id']}\nConfidence: {out['final_confidence']}\n\n"
        txt = header + "Final Answer:\n\n" + out['final_text'] + "\n\nTrace:\n"
        for t in out['trace']:
            txt += f"- [{t['ts']}] {t['msg']}\n"
        txt_path = OUT / f"{name}.txt"
        txt_path.write_text(txt, encoding='utf-8')
        json_path = OUT / f"{name}_full.json"
        json.dump(out, open(json_path,'w',encoding='utf-8'), indent=2, ensure_ascii=False)
        print("Saved:", txt_path)

if __name__ == '__main__':
    main()
