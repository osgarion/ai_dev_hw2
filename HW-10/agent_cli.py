import argparse
import sys
from pathlib import Path
from typing import List

from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever


class PdfIndex:
    """Load a PDF once and expose small helper tools."""

    def __init__(self, pdf_path: str, k: int = 4) -> None:
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        loader = PyPDFLoader(str(self.pdf_path))
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        chunks = splitter.split_documents(pages)

        self.pages = pages
        self.retriever = BM25Retriever.from_documents(chunks)
        self.retriever.k = k

    def preview(self) -> str:
        first = self.pages[0].page_content[:600] if self.pages else "(empty pdf)"
        return (
            f"File: {self.pdf_path.name}\n"
            f"Pages: {len(self.pages)}\n"
            f"First page excerpt:\n{first}"
        )

    def search(self, query: str) -> str:
        if not query.strip():
            return "Provide a non-empty query."

        hits = self.retriever.invoke(query)
        if not hits:
            return "No matching passages found."

        formatted: List[str] = []
        for idx, doc in enumerate(hits, start=1):
            page = doc.metadata.get("page")
            page_num = page + 1 if isinstance(page, int) else page
            content = " ".join(doc.page_content.split())
            snippet = content[:800] + ("..." if len(content) > 800 else "")
            formatted.append(f"[{idx}] page {page_num}: {snippet}")

        return "\n\n".join(formatted)

    def get_page(self, page_number: str) -> str:
        try:
            page_idx = int(page_number) - 1
        except ValueError:
            return "Page must be an integer (1-based)."

        if page_idx < 0 or page_idx >= len(self.pages):
            return f"Page out of range. PDF has {len(self.pages)} pages."

        content = " ".join(self.pages[page_idx].page_content.split())
        return f"Page {page_idx + 1}: {content[:1500]}"


def build_agent(index: PdfIndex):
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    def _preview(_in: str | None = None) -> str:
        return index.preview()

    def _search(query: str) -> str:
        return index.search(query)

    def _get_page(page_number: str) -> str:
        return index.get_page(page_number)

    tools = [
        Tool(
            name="preview_pdf",
            func=_preview,
            description="Use to get a quick overview of the PDF before deeper search (file name, pages, first page excerpt).",
        ),
        Tool(
            name="search_pdf",
            func=_search,
            description="Search the PDF for passages relevant to a natural language query; returns snippets with page numbers.",
        ),
        Tool(
            name="get_page",
            func=_get_page,
            description="Fetch the text of a specific page number (1-based). Helpful when you know which page to inspect.",
        ),
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        memory=memory,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF QA agent using LangChain + Ollama (Llama3).")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument("--question", help="Question for the agent (English).")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive session so the agent remembers prior turns.")
    parser.add_argument("--k", type=int, default=4, help="How many passages to retrieve for each search tool call.")

    args = parser.parse_args()

    try:
        index = PdfIndex(args.pdf, k=args.k)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - defensive for CLI
        print(f"Failed to load PDF: {exc}")
        return 1

    agent = build_agent(index)

    if args.interactive:
        print("Interactive mode. Type 'exit' or empty line to quit.\n")
        while True:
            question = input("Q: ").strip()
            if not question or question.lower() in {"exit", "quit", "q"}:
                break
            result = agent.invoke({"input": question})
            output = result.get("output") if isinstance(result, dict) else result
            print(f"A: {output}\n")
        return 0

    if not args.question:
        print("Please provide --question for one-shot mode or use --interactive.")
        return 1

    print("Running agent...\n")
    result = agent.invoke({"input": args.question})
    output = result.get("output") if isinstance(result, dict) else result
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
