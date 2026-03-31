import os
from pathlib import Path
from typing import List

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _load_local_env() -> None:
    """
    Carrega sempre o `.env` local da raiz do projeto (modo local).
    """
    try:
        from dotenv import load_dotenv

        dotenv_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(dotenv_path=dotenv_path, override=True)
    except Exception:
        pass


def _resolve_local_vector_path(db_name: str) -> Path:
    root_candidates = [
        Path(_env("DATA_PATH", "data")) / "processed",
        Path("data/processed"),
    ]
    for root in root_candidates:
        candidate = root / db_name
        if candidate.exists():
            return candidate
    return root_candidates[0] / db_name


def _list_available_vector_dbs() -> List[str]:
    dbs = set()
    roots = [
        Path(_env("DATA_PATH", "data")) / "processed",
        Path("data/processed"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for item in root.iterdir():
            if not item.is_dir():
                continue
            if (item / "index.faiss").exists() and (item / "index.pkl").exists():
                dbs.add(item.name)
    return sorted(dbs)


@st.cache_resource(show_spinner=False)
def load_vectorstore(db_name: str) -> FAISS:
    local_dir = _resolve_local_vector_path(db_name)
    if not local_dir.exists():
        available = _list_available_vector_dbs()
        available_text = ", ".join(available) if available else "nenhuma"
        raise RuntimeError(
            f"Vector store local não encontrado em '{local_dir}'. "
            f"Bases disponíveis: {available_text}. "
            f"Se necessário, gere a base com: python main.py --db-name {db_name}"
        )
    if not (local_dir / "index.faiss").exists() or not (local_dir / "index.pkl").exists():
        raise RuntimeError(
            f"Base '{db_name}' encontrada em '{local_dir}', mas está incompleta. "
            "Arquivos esperados: index.faiss e index.pkl."
        )

    embedding_model = _env("EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=embedding_model)

    return FAISS.load_local(
        str(local_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _format_context(docs: List) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        source = (doc.metadata or {}).get("source", "desconhecido")
        page = (doc.metadata or {}).get("page")
        page_info = f" (p. {page})" if page is not None else ""
        lines.append(f"[{i}] {doc.page_content}\nFonte: {source}{page_info}")
    return "\n\n".join(lines)


def _format_sources(docs: List) -> str:
    seen = set()
    lines = []
    for doc in docs:
        source = (doc.metadata or {}).get("source", "desconhecido")
        page = (doc.metadata or {}).get("page")
        label = f"{source} (p. {page})" if page is not None else source
        if label in seen:
            continue
        seen.add(label)
        lines.append(f"- {label}")
    return "\n".join(lines)


def _format_history(messages: List[dict], max_turns: int) -> str:
    if max_turns <= 0:
        return ""
    history = messages[-max_turns * 2 :]
    lines = []
    for msg in history:
        role = "Usuário" if msg["role"] == "user" else "Assistente"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _retrieve_docs(retriever, prompt: str) -> List:
    # LangChain novo: retriever.invoke(prompt)
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(prompt)
        return docs if isinstance(docs, list) else [docs]
    # Compat com versões antigas
    return retriever.get_relevant_documents(prompt)


def main() -> None:
    st.set_page_config(page_title="RAG Chat", page_icon="🔎", layout="wide")
    st.title("RAG Chatbot")

    _load_local_env()
    if not _env("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY não configurado no arquivo .env da raiz do projeto.")
        st.stop()

    with st.sidebar:
        st.header("Configuração")
        db_default = _env("VECTOR_DB_NAME", "vector_db") or "vector_db"
        dbs = _list_available_vector_dbs()
        if dbs:
            db_index = dbs.index(db_default) if db_default in dbs else 0
            db_name = st.selectbox("Base vetorial", options=dbs, index=db_index)
        else:
            db_name = st.text_input("Base vetorial (db_name)", value=db_default)
        k = st.slider("Documentos recuperados (k)", min_value=2, max_value=10, value=4)
        model = st.text_input("Modelo Chat", value=_env("CHAT_MODEL", "gpt-4o-mini") or "gpt-4o-mini")
        history_turns = st.slider(
            "Turnos de histórico",
            min_value=0,
            max_value=6,
            value=3,
            help=(
                "Quantos pares de conversa (usuário + assistente) entram no prompt. "
                "0 = sem histórico; 1 = última pergunta/resposta; 3 = últimas 3."
            ),
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Pergunte sobre os documentos...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando contexto..."):
            vectorstore = load_vectorstore(db_name)
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = _retrieve_docs(retriever, prompt)
            context = _format_context(docs)
            sources = _format_sources(docs)

        llm = ChatOpenAI(model=model)

        history_text = _format_history(st.session_state.messages[:-1], history_turns)
        prompt_text = f"""
Responda à pergunta com base apenas no contexto obtido a seguir e inclua a fonte utilizada como referência ao final.
Se houver histórico da conversa, use apenas para manter coerência, mas não invente fatos fora do contexto.

Histórico (se houver):
{history_text}

{context}

Question: {prompt}
"""
        response = llm.invoke(prompt_text)
        answer = response.content if hasattr(response, "content") else str(response)

        if sources:
            answer = f"{answer}\n\n**Fontes**\n{sources}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
