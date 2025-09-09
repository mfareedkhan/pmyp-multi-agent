## PMYP Multi-Agent Chat System (packaged)

#### **How to run (Colab / local):**
1. Ensure dependencies are installed (use requirements.txt).
2. Place this repo under '/content/pmyp_multi_agent' if using the current Colab workspace, or clone on a local machine.
3. Run:
   python run_scenarios.py

#### **What is included:**
- vector_store/inmemory.py
- agents/*.py (memory, research, analysis)
- coordinator.py (CoordinatorV3)
- run_scenarios.py (runs the five sample scenarios and writes outputs/)
- Dockerfile / docker-compose.yml
- requirements.txt

**Notes:**
- The code uses sentence-transformers 'all-MiniLM-L6-v2' for embeddings (CPU-friendly).
- Docker is optional; you can run directly in Colab or locally.
