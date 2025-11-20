# HW-10 PDF QA Agent

Simple CLI agent using LangChain + local Llama3 (Ollama) to answer questions over a PDF. Tools: preview PDF, search relevant passages (BM25), get page snippets.

## Setup
1) Create venv:
```
python -m venv .venv
.\.venv\Scripts\activate
```
2) Install deps (LangChain 0.3.x stack):
```
pip install -r requirements.txt
```
3) Ensure model is ready (downloads if missing). Default uses a smaller model for speed:
```
ollama pull llama3.1:8b
```

## Run (one-shot)
```
python agent_cli.py --pdf path\to\doc.pdf --question "Your question here"
```

## Run (interactive, remembers chat history)
```
python agent_cli.py --pdf path\to\doc.pdf --interactive
```

The agent loads the PDF, builds a BM25 retriever, and the Llama3 model decides how to use the tools (`preview_pdf`, `search_pdf`, `get_page`) to gather context and answer in English. In interactive mode it keeps conversation history within the session. The agent is implemented with LangChain's ReAct (`create_react_agent`).
