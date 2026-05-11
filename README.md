## Installation and Run Instructions

### 1. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

---

### 2. Configure Environment Variables

Then edit `.env` and add a valid Aliyun Bailian API key:

```env
OPENAI_API_KEY=your_bailian_api_key_here
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

MODEL_NAME=qwen-plus
EMBEDDING_MODEL=text-embedding-v4
VISION_MODEL_NAME=qwen-vl-plus

CHROMA_DIR=./chroma_db
COLLECTION_NAME=java_programming_kb
```


---

### 3. Build the Vector Database

Run the ingestion script:

```bash
python src/ingest.py
```

This processes the files in `data/raw/`, creates chunks, generates embeddings, and stores them in the Chroma vector database.

---

### 4. Run the Final Agent

Run the interactive assistant:

```bash
python main.py
```

Or ask a single question directly:

```bash
python main.py --question "What is the difference between ArrayList and LinkedList?"
```

---

### 5. Run Evaluation

Run the evaluation script:

```bash
python src/evaluation.py
```

The evaluation results will be saved to:

```text
outputs/eval_results.csv
outputs/eval_summary.json
```