import streamlit as st
import httpx
import json
import asyncio

from utils import StdOutHandler, interleave_lists


SEARCH_URL = "https://kagi-ml-dev-challenge.onrender.com/search"
SEARCH_LLM_URL = "https://kagi-ml-dev-challenge.onrender.com/search_llm"
ARTICLES_PER_PAGE = 20


st.set_page_config(page_title="Hacker News Search", page_icon="🔍")

account = "[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label=GiuseppeBellamacina&color=blue)](https://github.com/GiuseppeBellamacina)"
st.markdown(account, unsafe_allow_html=True)

st.title("🔍 Hacker News Relevance Search")
st.write("Enter your bio to find the most relevant Hacker News stories.")


user_bio = st.text_area(
    "✍️ **Your Bio**", placeholder="Describe your interests...", height=175
)
use_llm = st.toggle("🧠 **Use LLM for query expansion**", value=True)
k_value = st.slider(
    "🎯 **Number of results (k)**", min_value=10, max_value=500, value=500, step=10
)


if "results" not in st.session_state:
    st.session_state.results = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "duplicate_results" not in st.session_state:
    st.session_state.duplicate_results = 0


async def search_with_streaming():
    if not user_bio.strip():
        st.warning("⚠️ **Please enter a bio.**")
        return
    with st.spinner("🔍 Searching..."):
        try:
            col1, col2 = st.columns([1.4, 2])
            llm_container = col1.empty()
            results_container = col2.empty()
            latency_container = st.empty()
            errors_container = st.empty()

            handler = StdOutHandler(debug=True)
            handler.start(
                {
                    "llm": llm_container,
                    "results": results_container,
                    "latency": latency_container,
                    "errors": errors_container,
                }
            )

            st.session_state.results = []
            st.session_state.duplicate_results = 0

            async with httpx.AsyncClient() as client:
                if use_llm:
                    await handle_llm_search(client, handler)
                else:
                    await handle_standard_search(client, handler)
            if st.session_state.results:
                st.success(
                    f"🎉 **Search complete! {len(st.session_state.results)} results found.**"
                )
                if st.session_state.duplicate_results > 0:
                    st.info(
                        f"ℹ️ **Note:** {st.session_state.duplicate_results} duplicate results were removed."
                    )
            else:
                st.warning("⚠️ **No results found. Please try a different bio.**")
        except Exception as e:
            st.error(f"⚠️ **Connection error:** {str(e)}")
            raise e


async def handle_llm_search(client: httpx.AsyncClient, handler: StdOutHandler):
    async with client.stream(
        "POST",
        SEARCH_LLM_URL,
        json={"user_input": user_bio, "k": k_value},
        timeout=120.0,
    ) as response:

        results = []

        async for line in response.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                line = line[len("data: ") :]
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "error" in data:
                handler.error(Exception(data["error"]))
                return
            elif "chunk" in data:
                handler.on_new_chunk(data["chunk"])
            elif "results" in data and "query" in data:
                handler.on_new_results(data["query"], data["results"])
                results.append(data["results"])
            elif "done" in data:
                break
            else:
                handler.error(Exception(f"Invalid data: {data}"))
                return
            
        number_of_results = sum(len(r) for r in results)
        st.session_state.results = interleave_lists(results)
        st.session_state.duplicate_results = (
            number_of_results - len(st.session_state.results)
        )


async def handle_standard_search(client: httpx.AsyncClient, handler: StdOutHandler):
    response = await client.post(
        SEARCH_URL, json={"user_input": user_bio, "k": k_value}, timeout=60.0
    )

    if response.status_code != 200:
        handler.error(Exception(f"Error {response.status_code}: {response.text}"))
        return
    data = response.json()

    if "error" in data:
        handler.error(Exception(data["error"]))
        return
    if "results" in data:
        handler.on_new_results(user_bio, data["results"], show_latency=True)

        st.session_state.results = data["results"]


if st.button("🔎 **Search**", use_container_width=True):
    asyncio.run(search_with_streaming())
if st.session_state.results:
    total_results = len(st.session_state.results)
    total_pages = (total_results + ARTICLES_PER_PAGE - 1) // ARTICLES_PER_PAGE
    page = st.session_state.current_page

    st.markdown(f"### 📄 Page {page} of {total_pages}")

    start_idx = (page - 1) * ARTICLES_PER_PAGE
    end_idx = min(start_idx + ARTICLES_PER_PAGE, total_results)

    page_results = st.session_state.results[start_idx:end_idx]

    for story in page_results:
        title = story["title"]
        url = story["url"]
        with st.container():
            st.markdown(
                f"""
                <div style="padding: 10px; border-radius: 10px; background-color: #191919; margin-bottom: 10px;">
                    <h4 style="margin-bottom: 5px;">🔗 <a href="{url}" target="_blank" style="color: #FFB319;">{title}</a></h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if page > 1 and st.button("⬅️ Previous Page"):
            st.session_state.current_page -= 1
            st.rerun()
    with col3:
        if page < total_pages and st.button("Next Page ➡️"):
            st.session_state.current_page += 1
            st.rerun()
