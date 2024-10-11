# Setup

Need a minimum Neo4j version of 5.15 installed either locally or in the cloud.

Update the DB_CONFIG in microsoft_to_neo4j.py with your Neo4j details.

Install the required python packages:

```bash
pip install -r requirements.txt
```

## Ingest Data Into Microsoft Graph Rag Locally

Create a folder called ragtest in the root directory and copy your data into it:

Windows:
```powershell
if not exist "./ragtest/input" mkdir "./ragtest/input"
copy "book.txt" "./ragtest/input/book.txt"
python -m graphrag.index --init --root ./ragtest
```

Linux:
```bash
mkdir -p ragtest/input
cp book.txt ragtest/input/
python -m graphrag.index --init --root ./ragtest
```

## Environment
GRAPHRAG_LLM_MODEL=gpt-4o-mini
GRAPHRAG_ENTITY_EXTRACTION_ENTITY_TYPES=organization,person,event,geo
GRAPHRAG_ENTITY_EXTRACTION_MAX_GLEANINGS=1
GRAPHRAG_CLAIM_EXTRACTION_ENABLED=False
GRAPHRAG_CLAIM_EXTRACTION_MAX_GLEANINGS=1
OPENAI_API_KEY and GRAPHRAG_API_KEY are also required and should be set to your openai key.

All required environment variables are set in the .env file.

## References

https://neo4j.com/developer-blog/microsoft-graphrag-neo4j/
