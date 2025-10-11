# Bot Ice Breaker AI

An AI-powered resume analysis and interactive chatbot that ingests a candidate's PDF resume, builds a semantic vector index of the content, and lets recruiters ask detailed follow-up questions. The application uses Gradio for the UI, LlamaIndex for retrieval-augmented generation (RAG), and OpenRouter + HuggingFace for flexible LLM / embedding model access.

---

## 🚀 Why LlamaIndex?

LlamaIndex is chosen as the orchestration layer for Retrieval Augmented Generation because it:

- **Simplifies RAG pipelines**: Provides high-level abstractions (indexing, retrievers, query engines) so we focus on logic, not plumbing.
- **Flexible data connectors**: Easy to convert structured/unstructured resume text into nodes and store/query them.
- **Composable retrieval**: Allows adjusting `similarity_top_k` and embedding strategies without rewriting core logic.
- **Model-agnostic**: Works smoothly with different LLM backends (HuggingFace, OpenRouter, LangChain wrappers).
- **Prompt customization**: Supports custom `PromptTemplate` objects used here to tailor candidate analysis and Q&A.

In this project, LlamaIndex handles:

1. Chunking resume data (done manually in `data_processing.py`).
2. Building a `VectorStoreIndex` from processed nodes.
3. Retrieving the most relevant chunks for each user query.
4. Supplying retrieved context + prompt template to the LLM for grounded answers.

---

## 🔑 Why OpenRouter?

OpenRouter is used as the LLM gateway because it:

- **Aggregates multiple frontier models** (DeepSeek, Anthropic, OpenAI, Mistral, etc.) under a unified API.
- **Flexible model selection**: Swap `OPENROUTER_MODEL` to experiment without changing code.
- **Cost & performance optimization**: Choose cheaper or faster models per deployment environment.
- **Failover potential**: Can implement fallback logic if one model becomes unavailable.
- **Future-proofing**: Avoid lock-in to a single vendor.

The app can also use HuggingFace hosted models for embeddings and (optionally) generation, combining the strengths of both ecosystems.

---

## 🧱 Project Structure

```
.
├── app.py                     # Gradio interface (main entry point for UI)
├── main.py                    # CLI-style processing example
├── config.py                  # Centralized settings via pydantic BaseSettings
├── module/
│   ├── extract_profile_pdf.py # PDF text extraction logic
│   ├── data_processing.py     # Splitting, node creation, vector index, verification
│   ├── query_engine.py        # Fact generation + user Q&A using LlamaIndex
│   ├── llm_interface.py       # Model factory (LLM + embeddings via OpenRouter / HF)
├── requirements.txt           # Base (un-pinned) dependencies
├── requirements.prod.txt      # Pinned production dependencies
├── Dockerfile                 # Multi-stage (builder + production) image
├── docker-compose.yml         # Dev compose (live bind mount)
├── docker-compose.yml    # Hardened production compose
├── .env.example               # Example environment variables
└── README.md                  # This file
```

---

## 🏗 Arsitektur Aplikasi (FastAPI + Gradio)

Gradio UI sekarang di-_mount_ pada instance **FastAPI** menggunakan `gr.mount_gradio_app(app, create_gradio_interface(), path="/")`, sehingga kita dapat:

- Menambah endpoint REST (mis. `/healthz`, `/metrics`, `/sessions`).
- Menjalankan server produksi dengan **Uvicorn/Gunicorn**.
- Integrasi mudah dengan reverse proxy (Nginx / Caddy / Traefik) untuk TLS & header keamanan.

Alur utama:

1. Upload PDF → `extract_profile_pdf.py` mengekstrak teks.
2. Split teks → `data_processing.py` membuat potongan (nodes) dan index vektor.
3. Prompt awal → `generate_facts_candidate()` menghasilkan ringkasan kandidat.
4. Pertanyaan user → `answer_user_question()` melakukan retrieval (RAG) + jawaban kontekstual.
5. Gradio di-_mount_ pada root path FastAPI (`/`).

---

## 🔀 Mode Menjalankan

| Mode              | Perintah                                      | Kapan digunakan             | Catatan                             |
| ----------------- | --------------------------------------------- | --------------------------- | ----------------------------------- |
| Gradio langsung   | `python app.py`                               | Eksplorasi cepat / debug UI | Tidak perlu FastAPI, fitur terbatas |
| FastAPI + Uvicorn | `uvicorn app:app --host 0.0.0.0 --port 7860`  | Produksi ringan             | Dapat tambah endpoint lain          |
| Docker (dev)      | `docker compose up -d`                        | Lingkungan terisolasi       | Port 7860 dipetakan ke host         |
| Docker + Nginx    | `docker compose up -d` (dengan service nginx) | Reverse proxy, siap TLS     | Akses melalui port 80 / 443         |

> Gunakan Uvicorn untuk performa & kontrol. Tambahkan `--workers 2` jika beban mulai naik (hindari terlalu banyak worker bila model embedding/LLM berat).

---

## ▶️ Jalankan Lokal (FastAPI + Uvicorn)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # isi token & API keys
uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
```

Akses: http://localhost:7860

Endpoint custom (opsional) contoh:

```python
# Tambahkan sebelum mount gradio di app.py
@app.get("/healthz")
def health():
		return {"status": "ok"}
```

---

## 🐳 Docker (FastAPI + Uvicorn)

Compose menjalankan:

```yaml
command: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

Build & run:

```bash
docker compose build
docker compose up -d
docker compose logs -f bot-ice-breaker
```

Tes import uvicorn:

```bash
docker compose exec bot-ice-breaker python -c "import uvicorn; print('ok')"
```

---

## 🩺 Health Check & Observabilitas

Docker healthcheck default melakukan `curl` ke root. Disarankan endpoint khusus `/healthz` agar jelas status aplikasi. Tambahkan juga metric endpoint (misal dengan Prometheus FastAPI middleware) pada fase lanjut.

---

## 🛡 Keamanan Tambahan

- Jangan log isi lengkap resume (PII). Sanitasi atau ringkas.
- Gunakan `read_only: true` + non-root user (sudah di Dockerfile/compose).
- Set `share=False` di Gradio untuk mencegah tunneling publik otomatis.
- Rate limiting bisa ditambah via Nginx (`limit_req_zone`).
- Pertimbangkan Web Application Firewall jika dipublikasikan.

---

## 🔧 Variabel Lingkungan Tambahan

| Variabel    | Fungsi                                                 |
| ----------- | ------------------------------------------------------ |
| `BIND_HOST` | Override host binding (default `0.0.0.0` di container) |

---

## ⚠️ Debug Umum

| Masalah                        | Penyebab                                | Solusi                                            |
| ------------------------------ | --------------------------------------- | ------------------------------------------------- |
| `ModuleNotFoundError: uvicorn` | Uvicorn belum terinstall dalam image    | Pastikan ada di `requirements.txt`, rebuild image |
| Tidak bisa akses dari browser  | Server bind ke `127.0.0.1` di container | Pastikan host = `0.0.0.0` / override `BIND_HOST`  |
| Respon lambat                  | Chunk besar / top_k terlalu tinggi      | Kurangi `CHUNK_SIZE` / `SIMILARITY_TOP_K`         |
| API Key invalid                | Token salah / belum di-set              | Revisi `.env` dan rebuild                         |

---

## 🔄 Update Production Singkat

```bash
git pull
docker compose build --no-cache
docker compose up -d --force-recreate
```

---

## ✅ Quickstart (Ringkas)

```bash
# Lokal (FastAPI + Gradio)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # isi API keys
uvicorn app:app --host 0.0.0.0 --port 7860
```

```bash
# Docker
cp .env.example .env
docker compose up -d --build
```

```bash
# Tambah Nginx (optional)
# Pastikan nginx/nginx.conf ada lalu tambahkan service nginx di compose
```

---

## ⚙️ Core Workflow

1. Upload a resume PDF via Gradio UI.
2. Extract raw text using `extract_profile_pdf.py`.
3. Split and preprocess into chunks / nodes (`data_processing.py`).
4. Build a `VectorStoreIndex` and verify embedding model.
5. Generate initial structured facts ("candidate overview") using a custom prompt.
6. Accept user questions; retrieve relevant chunks (`similarity_top_k`) and answer via LLM with grounded context.

---

## 📦 Dependencies

- `llama-index` + related subpackages (core, embeddings, LLM bridges)
- `langchain` / `langchain-community` for interoperability
- `gradio` for UI
- `pydantic-settings` for configuration management
- `requests`, `uuid`, etc. for utilities

Production adds pinned versions + optional servers like `gunicorn` or `uvicorn` (should you wrap in FastAPI later).

---

## 🔐 Environment Variables (`.env`)

| Variable                      | Required | Default                                | Description                              |
| ----------------------------- | -------- | -------------------------------------- | ---------------------------------------- |
| `PORT`                        | No       | 7860                                   | Web UI port                              |
| `HUGGINGFACE_MODEL_EMBEDDING` | No       | sentence-transformers/all-MiniLM-L6-v2 | Embedding model for vector index         |
| `HUGGINGFACE_TOKEN`           | Yes      | (none)                                 | HuggingFace access token                 |
| `OPENROUTER_API_KEY`          | Yes      | (none)                                 | API key for OpenRouter models            |
| `OPENROUTER_MODEL`            | No       | deepseek/deepseek-chat-v3.1            | Default LLM model via OpenRouter         |
| `TOP_K`                       | No       | 5                                      | Model sampling parameter (if applicable) |
| `TOP_P`                       | No       | 0.95                                   | Nucleus sampling (if applicable)         |
| `MAX_NEW_TOKENS`              | No       | 512                                    | Max tokens for answer generation         |
| `MIN_NEW_TOKENS`              | No       | 256                                    | Min tokens (if supported)                |
| `TEMPERATURE`                 | No       | 0.1                                    | Creativity setting                       |
| `CHUNK_SIZE`                  | No       | 400                                    | Resume chunk length                      |
| `SIMILARITY_TOP_K`            | No       | 7                                      | Number of retrieved context chunks       |

Create your own `.env` from `.env.example`.

---

## 🖥️ Running Locally (Host Machine)

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd bot-ice-breaker-ai
cp .env.example .env
# Edit .env and insert your API keys (OPENROUTER_API_KEY, HUGGINGFACE_TOKEN, etc.)
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the App

```bash
python app.py
```

Access the UI at: `http://127.0.0.1:7860`

### 4. (Optional) Use `main.py` for CLI Testing

```bash
python main.py /path/to/resume.pdf
```

---

## 🐳 Running with Docker (Development)

### 1. Build & Start (Dev Compose)

```bash
# From repo root
docker compose build
docker compose up -d
```

Visit: `http://localhost:7860`

### 2. Live Code Changes

The dev compose mounts the working directory so edits reflect instantly. Restart if dependency changes:

```bash
docker compose restart web
```

### 3. Stop

```bash
docker compose down
```

---

## 🛡️ Production Deployment (Docker)

Two approaches: **single host** (Docker + docker-compose) or **container registry + orchestrator** (e.g. ECS, Kubernetes). This README covers single host.

### 1. Use Production Dockerfile

`Dockerfile` (multi-stage) or `Dockerfile.prod` depending on how you've named the finalized version. Current repo uses multi-stage already.

### 2. Build Production Image

```bash
docker build -t bot-ice-breaker-ai:prod -f Dockerfile .
```

### 3. Prepare Environment

```bash
cp .env.example .env
# Edit .env with real keys
mkdir -p logs temp
```

### 4. Run with Production Compose

```bash
docker compose -f docker-compose.yml up -d --build
```

Access at: `http://<server-ip>:7860`

### 5. Check Health / Logs

```bash
docker compose -f docker-compose.yml ps
curl -f http://localhost:7860/ || echo "Health endpoint not responding yet"
docker compose -f docker-compose.yml logs -f
```

### 6. Update / Redeploy

```bash
git pull origin main
docker compose -f docker-compose.yml build --no-cache
docker compose -f docker-compose.yml up -d
```

### 7. Stop / Clean Up

```bash
docker compose -f docker-compose.yml down
# Remove volumes (careful!)
docker compose -f docker-compose.yml down -v --remove-orphans
```

---

## 🔍 Security & Privacy Considerations

- Resumes contain PII—never persist raw text unless required.
- Consider encrypting logs or disabling logging of raw resume content.
- Rotate API keys regularly.
- Add request rate limiting if exposed publicly.

---

## 📈 Scaling Paths

| Component    | Strategy                                             |
| ------------ | ---------------------------------------------------- |
| Gradio App   | Put behind Nginx reverse proxy / Caddy with HTTPS    |
| LLM Calls    | Batch / cache responses; consider model distillation |
| Vector Index | Persist to external store (e.g., Chroma, PGVector)   |
| Multi-User   | Map `session_id` to secure storage (Redis)           |
| Monitoring   | Add Prometheus exporters / structured JSON logs      |

---

## 🧪 Testing Ideas (Not Yet Implemented)

- Unit tests for text splitting edge cases.
- Mocked LLM responses for deterministic tests.
- PDF parsing fallback strategies.
- Performance benchmarks (token counts, latency).

---

## 🛠️ Future Improvements

- Add authentication layer for recruiter access.
- Persistent vector store (e.g., Milvus / Weaviate).
- Automatic model fallback (OpenRouter multi-model sequence).
- Add metadata extraction (skills, years of experience tagging).

---

## ❓ FAQ

**Q: Can I swap models easily?**  
Yes—change `OPENROUTER_MODEL` in `.env`.

**Q: Do I need both HuggingFace and OpenRouter?**  
Not strictly. You can use one for embeddings and/or generation. This dual approach keeps flexibility.

**Q: Why not just LangChain?**  
LangChain is used indirectly for some integrations; LlamaIndex provides simpler abstractions for indexing + querying resume text with less boilerplate.

---

## ▶️ Quickstart Summary

```bash
# Local quickstart
git clone <repo>
cd bot-ice-breaker-ai
cp .env.example .env  # add API keys
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py  # visit http://127.0.0.1:7860
```

```bash
# Production (Docker)
cp .env.example .env  # add API keys
mkdir -p logs temp
docker build -t bot-ice-breaker-ai:prod -f Dockerfile .
docker compose -f docker-compose.yml up -d --build
```

---

## 📝 License

This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

---

**Enjoy exploring candidates intelligently with Bot Ice Breaker AI!**
