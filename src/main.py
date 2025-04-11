import os
import asyncio
import json

from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.responses import StreamingResponse, JSONResponse
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate import WeaviateVectorStore
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.utils import Request, CustomHFEmbeddings
from src.prompt import make_prompt


EMBEDDER_ENDPOINT_URL = os.environ["EMBEDDER_ENDPOINT_URL"]
WEAVIATE_INDEX_NAME = os.environ["WEAVIATE_INDEX_NAME"]


chain = embedder = client = db = None


# FastAPI lifespan event handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain, embedder, client, db

    ### LLM configuration ###

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = make_prompt()
    chain = prompt | model
    print("Chain created")

    ### Embedder configuration ###
    
    embedder = CustomHFEmbeddings(EMBEDDER_ENDPOINT_URL)
    print("Embedder created")

    ### WEAVIATE cliet configuration ###

    weaviate_url = os.environ["WEAVIATE_URL"]
    weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )

    if not client.is_ready():
        print("Weaviate is not ready yet")
    else:
        print("Weaviate is ready")
    db = WeaviateVectorStore(
        client=client,
        index_name=WEAVIATE_INDEX_NAME,
        text_key="text",
        embedding=embedder,
    )
    print("All set up")

    yield

    client.close()


# FastAPI app

app = FastAPI(lifespan=lifespan)


def serialize_docs(docs: list[Document]) -> list[dict]:
    return [
        {
            "title": doc.page_content,
            "url": doc.metadata.get("url", ""),
            "hn_id": doc.metadata.get("hn_id", ""),
        }
        for doc in docs
    ]


async def wait_for_db_ready(timeout=30) -> bool:
    for _ in range(timeout):
        if client.is_ready():
            return True
        await asyncio.sleep(1)
    return False


async def search_stories(query, k) -> list[dict]:
    if not await wait_for_db_ready():
        return []
    results = await db.asimilarity_search(query, k)
    return serialize_docs(results)


# Endpoint for simple search (No LLM)


@app.post("/search")
async def search(request: Request):
    query = request.user_input
    k = request.k

    if not query:
        return JSONResponse({"error": "Missing query"}, status_code=400)
    results = await search_stories(query, k)
    return JSONResponse({"results": results})


# Endpoint for inactivity check


@app.get("/keep_alive")
async def keep_alive():
    # A simple endpoint to keep the server alive and prevent it from sleeping.
    return {"message": "I don't want to spend money on a better hosting plan"}


async def generate_results(query, k):
    """
    Generate a streaming response for the given query.

    This function performs the following steps:
      1. Streams tokens from the LLM and immediately sends each token (as 'chunk') to the client.
      2. Accumulates tokens into a buffer and extracts subqueries whenever a newline ("\n") is found.
      3. After the LLM streaming is complete, performs a database search for each extracted subquery.
         The number of results per query is computed as k divided by the number of subqueries (with a minimum of 1).
      4. Sends the serialized documents for each subquery as a single event.
      5. If a database error occurs, sends an error message for that subquery but continues processing.
      6. Finally, signals to the client that streaming is complete.

    Parameters:
      query (str): The user query to process.
      k (int): The total number of results expected; divided among subqueries.

    Yields:
      str: Server-Sent Events (SSE) formatted strings containing chunks, results, errors, or a done message.
    """
    buffer = ""
    sub_queries = []

    # LLM streaming: immediately sends each token to the client.
    async for token_obj in chain.astream({"input": query}):
        token = token_obj.content
        if token:
            # Immediately send the token
            yield f"data: {json.dumps({'chunk': token})}\n\n"
            buffer += token

        # Extract subqueries from the buffer when a newline ("\n") is found.
        while "\n" in buffer:
            sub_query, buffer = buffer.split("\n", 1)
            sub_query = sub_query.strip()
            if sub_query:
                sub_queries.append(sub_query)

    if buffer:
        sub_query = buffer.strip()
        if sub_query:
            sub_queries.append(sub_query)

    # At this point, the LLM streaming has completed.
    # If any subqueries were generated, perform a DB search.

    if sub_queries:
        k_per_query = max(1, k // len(sub_queries))
        queue = asyncio.Queue() # Queue to store results and errors

        async def worker(sub_query):
            try:
                results = await search_stories(sub_query, k_per_query)
                await queue.put((sub_query, results, None))
            except Exception as e:
                await queue.put((sub_query, None, str(e)))

        # Launch all database search tasks concurrently
        tasks = [asyncio.create_task(worker(sub_query)) for sub_query in sub_queries]

        # Process results as they become available
        for _ in range(len(tasks)):
            sub_query, results, error = await queue.get()
            if error:
                yield f"data: {json.dumps({'error': f'DB error on {sub_query}: {error}'})}\n\n"
            else:
                yield f"data: {json.dumps({'results': results, 'query': sub_query})}\n\n"

        # Ensure all tasks are completed before proceeding
        await asyncio.gather(*tasks, return_exceptions=True)

    # Signal to the client that streaming is complete.
    yield 'data: {"done": true}\n\n'


# Endpoint for LLM search


@app.post("/search_llm")
async def search_llm(request: Request):
    query = request.user_input
    k = request.k

    if not query:
        return JSONResponse({"error": "Missing query"}, status_code=400)
    return StreamingResponse(
        generate_results(query, k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-transform"},
    )
