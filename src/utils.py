from time import time
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from itertools import zip_longest
from langchain_core.runnables import Runnable


class Request(BaseModel):
    user_input: str = Field(description="User's input")
    k: int = Field(default=500, description="Number of results to return")


class StdOutHandler:
    """
    Handles the output of the LLM and the results from the DB.
    """
    def __init__(self, debug=False):
        self.debug = debug
        self.containers = {}
        self.llm_text = ""
        self.results_text = ""
        self.first_token = True
        self.latency = 0

    def start(self, containers: dict):
        """
        Initialize the handler with the containers to update and reset the internal state.
        """
        self.containers = containers
        self.llm_text = ""
        self.results_text = ""
        self.latency = time()
        self.first_token = True

    def on_new_chunk(self, chunk: str) -> None:
        """Handles the new chunk of text from the LLM."""
        if not chunk:
            return

        if self.first_token:
            self.first_token = False
            self.latency = time() - self.latency
            if self.debug:
                print(f"[First token received after {self.latency:.2f} sec]")
            if "latency" in self.containers:
                latency_content = f"<p><b>⌛ Latency:</b> {self.latency:.2f} sec</p>"
                self.containers["latency"].markdown(latency_content, unsafe_allow_html=True)
        
        self.llm_text += chunk.replace("\n", "<br>")
        if "llm" in self.containers:
            content = (
            f"<h3>🔄 Query Expansion</h3>"
            f"<p>{self.llm_text}</p>"
            )
            self.containers["llm"].markdown(content, unsafe_allow_html=True)
        
        if self.debug:
            print(chunk, end="", flush=True)

    def on_new_results(self, query: str, results: list, show_latency: bool = False) -> None:
        """Handles the new results from the DB."""
        count = len(results)
        if count == 0:
            message = f"🔍 <i>{query}</i> ❌ <b>No results found.</b><br>"
        else:
            message = f"🔍 <i>{query}</i> ✅ <b>{count} results received.</b><br>"
        self.results_text += message
        if "results" in self.containers:
            content = f"<h3>🌐 Search Status</h3><p>{self.results_text}</p>"
            self.containers["results"].markdown(content, unsafe_allow_html=True)
        if show_latency and "latency" in self.containers:
            self.latency = time() - self.latency
            latency_content = f"<p><b>⌛ Latency:</b> {self.latency:.2f} sec</p>"
            self.containers["latency"].markdown(latency_content, unsafe_allow_html=True)
        if self.debug:
            print(message, flush=True)

    def error(self, error: Exception):
        """
        Handles an error during the processing of the LLM or the DB results.
        """
        if self.debug:
            print(f"\033[1;31m[STDOUTHANDLER ERROR]\033[0m: {error}")
        if "errors" in self.containers:
            self.containers["errors"].markdown(f"<h3>❌ Error</h3><p>{error}</p>", unsafe_allow_html=True)
        raise error


class ChainInterface(ABC):   
    @abstractmethod
    def invoke(self, input, containers=None):
        pass
    
    @abstractmethod
    def stream(self, input, containers=None):
        pass
    
    @abstractmethod
    async def ainvoke(self, input, containers=None):
        pass
    
    @abstractmethod
    async def astream(self, input, containers=None):
        pass


class Chain(ChainInterface):
    def __init__(self, runnable: Runnable, handler: StdOutHandler | None, name: str = "Chain"):
        self.runnable = runnable
        self.handler = handler
        self.name = name
    
    def invoke(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = self.runnable.invoke(input)
            if self.handler:
                self.handler.on_new_token(response)
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return ""
    
    async def ainvoke(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = await self.runnable.ainvoke(input)
            if self.handler:
                self.handler.on_new_token(response)
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return ""

    def stream(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = ""
            out = self.runnable.stream(input)
            for token in out:
                if self.handler:
                    self.handler.on_new_token(token)
                response += token.content
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return ""
    
    async def astream(self, input, containers=None):
        try:
            if self.handler:
                self.handler.start(containers)
            response = ""
            out = self.runnable.astream(input)
            async for token in out:
                if self.handler:
                    self.handler.on_new_token(token)
                response += token.content
            return response
        except Exception as e:
            if self.handler:
                self.handler.error(e)
            else:
                raise e
            return ""


def interleave_lists(lists):
    if not lists or not all(isinstance(l, list) for l in lists):
        return lists
    result = []
    for elements in zip_longest(*lists, fillvalue=None):
        result.extend(filter(lambda x: x is not None, elements))
    return result
