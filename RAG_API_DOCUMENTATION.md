# RAG System API Documentation

This guide details how to interact with the deployed RAG system API.

## Base URL
For internal server-to-server calls on the same VPS/Coolify host, use `http://localhost:8081`.
If you need the public domain instead, use `https://rag.buildomain.com`.

> [!WARNING]
> **Authentication Note**: The current API implementation **does not** enforce authentication (API keys) by default. It is recommended to configure Basic Auth or similar on your reverse proxy (Coolify/Nginx) or implement API key logic in the application if exposed publicly.

---

## Endpoints

### 1. Health Check
Verify the service is running.

- **URL**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy",
    "timestamp": "2024-03-20T10:00:00"
  }
  ```

### 2. Upload Document
Upload a PDF, DOCX, or TXT file to be indexed.

- **URL**: `/upload`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: The file object (Required).
  - `docid`: A unique ID for the document (Required).
  - `collection_name`: Name of the collection to store in (Default: `default_collection`).

**Example (Python)**:
```python
import requests

url = "http://localhost:8081/upload"
files = {
    'file': ('contract.pdf', open('docs/contract.pdf', 'rb'), 'application/pdf')
}
data = {
    'docid': 'doc_123',
    'collection_name': 'legal_docs'
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### 3. Query (Enhanced Hybrid)
Search your documents using RAG.

- **URL**: `/query_hybrid_enhanced`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "query": "What are the termination conditions?",
    "collection_name": "legal_docs",
    "top_k": 5,
    "llm": "gpt-4o",  # Optional: gpt-3.5-turbo, deepseek-chat, claude-3-opus
    "balance_emphasis": "high_precision" # Optional
  }
  ```

**Example (cURL)**:
```bash
curl -X POST http://localhost:8081/query_hybrid_enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the project timeline",
    "collection_name": "project_alpha"
  }'
```

### 4. Async Job Status
Check the status of long-running tasks.

- **URL**: `/job_status/<job_id>`
- **Method**: `GET`

---

## Python Client Example

Here is a simple Python wrapper to interact with your services:

```python
import requests

class RagClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')

    def health(self):
        return requests.get(f"{self.base_url}/health").json()

    def upload_document(self, file_path, doc_id, collection="default"):
        url = f"{self.base_url}/upload"
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'docid': doc_id, 'collection_name': collection}
            response = requests.post(url, files=files, data=data)
        return response.json()

    def query(self, question, collection="default"):
        url = f"{self.base_url}/query_hybrid_enhanced"
        payload = {
            "query": question,
            "collection_name": collection
        }
        response = requests.post(url, json=payload)
        return response.json()

# Usage
client = RagClient("http://localhost:8081")
print(client.health())
print(client.query("Summarize the Q3 report", collection="financials"))
```
