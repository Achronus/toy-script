# toy-script

Toy experiment for testing PydanticAI's [RAG](https://ai.pydantic.dev/examples/rag/) example with a non-OpenAI approach.

Seems to be broken. Agent cannot detect tools even with an extremely direct system prompt `Use the 'retrieve' tool to answer user queries`.

Link to main code file -> [rag.py](https://github.com/Achronus/toy-script/blob/main/toy_script/rag.py).

Commands to recreate issue:

```bash
mkdir postgres-data
docker run --rm -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d -v /postgres-data:/var/lib/postgresql/data pgvector/pgvector:pg17
```

```bash
python -m toy_script.rag build
python -m toy_script.rag search
```
