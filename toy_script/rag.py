import json
import asyncio
import re
import sys
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass

import asyncpg
import logfire

import pydantic_core
from pydantic import TypeAdapter
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.gemini import GeminiModel

from sentence_transformers import SentenceTransformer
from typing_extensions import AsyncGenerator

from toy_script.settings import AgentSettings


DOCS_JSON = "logfire_docs.json"
model = SentenceTransformer("all-MiniLM-L6-v2")
settings = AgentSettings()

logfire.configure()
logfire.instrument_asyncpg()


@dataclass
class Deps:
    pool: asyncpg.Pool


m = GeminiModel("gemini-1.5-flash", api_key=settings.GEMINI_API_KEY)
agent = Agent(
    m,
    deps_type=Deps,
    retries=3,
    system_prompt="Use the 'retrieve' tool to answer user queries.",
)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query.
    """
    with logfire.span(
        "create embedding for {search_query=}", search_query=search_query
    ):
        embedding = model.encode(search_query)

    embedding_json = pydantic_core.to_json(embedding.tolist()).decode()
    logfire.info("Embedding json:", embedding_json=embedding_json)

    rows = await context.deps.pool.fetch(
        "SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8",
        embedding_json,
    )
    return "\n\n".join(
        f"# {row['title']}\nDocumentation URL:{row['url']}\n\n{row['content']}\n"
        for row in rows
    )


async def run_agent(question: str):
    """Entry point to run the agent and perform RAG based question answering."""
    logfire.info('Asking "{question}"', question=question)

    async with database_connect(False) as pool:
        deps = Deps(pool=pool)

        with logfire.span("Agent response") as span:
            try:
                answer = await agent.run(question, deps=deps)
                span.set_attribute("result", answer.data)
                print(answer.data)
            except ValueError as e:
                span.record_exception(e)


async def build_search_db():
    """Build the search database."""
    with open(DOCS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_json = json.dumps(data)
    sections = sessions_ta.validate_json(data_json)

    async with database_connect(True) as pool:
        with logfire.span("create schema"):
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(DB_SCHEMA)

        sem = asyncio.Semaphore(10)
        async with asyncio.TaskGroup() as tg:
            for section in sections:
                tg.create_task(insert_doc_section(sem, pool, section))


@dataclass
class DocsSection:
    id: int
    parent: int | None
    path: str
    level: int
    title: str
    content: str

    def url(self) -> str:
        url_path = re.sub(r"\.md$", "", self.path)
        return (
            f"https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title, '-')}"
        )

    def embedding_content(self) -> str:
        return "\n\n".join((f"path: {self.path}", f"title: {self.title}", self.content))


sessions_ta = TypeAdapter(list[DocsSection])


async def insert_doc_section(
    sem: asyncio.Semaphore,
    pool: asyncpg.Pool,
    section: DocsSection,
) -> None:
    async with sem:
        url = section.url()
        exists = await pool.fetchval("SELECT 1 FROM doc_sections WHERE url = $1", url)

        if exists:
            logfire.info("Skipping {url=}", url=url)
            return

        with logfire.span("create embedding for {url=}", url=url):
            embedding = model.encode(section.embedding_content())

        embedding_json = pydantic_core.to_json(embedding.tolist()).decode()

        await pool.execute(
            "INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)",
            url,
            section.title,
            section.content,
            embedding_json,
        )


@asynccontextmanager
async def database_connect(
    create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
    server_dsn, database = (
        "postgresql://postgres:postgres@localhost:5432",
        "pydantic_ai_rag",
    )
    if create_db:
        with logfire.span("check and create DB"):
            conn = await asyncpg.connect(server_dsn)
            try:
                db_exists = await conn.fetchval(
                    "SELECT 1 FROM pg_database WHERE datname = $1", database
                )
                if not db_exists:
                    await conn.execute(f"CREATE DATABASE {database}")
            finally:
                await conn.close()

    pool = await asyncpg.create_pool(f"{server_dsn}/{database}")
    try:
        yield pool
    finally:
        await pool.close()


DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_sections (
    id serial PRIMARY KEY,
    url text NOT NULL UNIQUE,
    title text NOT NULL,
    content text NOT NULL,
    -- embedding model returns vector of 384
    embedding vector(384) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""


def slugify(value: str, separator: str, unicode: bool = False) -> str:
    """Slugify a string, to make it URL friendly."""
    # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
    if not unicode:
        # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
        value = unicodedata.normalize("NFKD", value)
        value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(rf"[{separator}\s]+", separator, value)


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else None
    if action == "build":
        asyncio.run(build_search_db())
    elif action == "search":
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = "How do I configure logfire to work with FastAPI?"
        asyncio.run(run_agent(q))
    else:
        print("Something went wrong.")
        sys.exit(1)
