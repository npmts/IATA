from llm_assistant import LLM_Assistant
from embedding_models import Embedding_models
from vector_store import Vector_stores
from chat_history import ChatHistory

import streamlit as st
import yaml
import pymupdf4llm
import tempfile
import os
from langchain.text_splitter import MarkdownTextSplitter
from PIL import Image
import requests
from io import BytesIO
from typing import Dict, List


# def stream_data_openai(stream_response):
#     full_response = ""
#     for chunk in stream_response:
#         try:
#             response = chunk.choices[0].delta.content
#             yield response
#             full_response += response
#             time.sleep(0.02)
#         except:
#             pass
#     return full_response


def get_texts(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    markdown_text = pymupdf4llm.to_markdown(temp_file_path)
    os.remove(temp_file_path)

    return markdown_text


def get_chunks(markdown_text, chunk_size, chunk_overlap):
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(markdown_text)
    clean_chunks = [str(chunk) for chunk in chunks if chunk is not None and str(chunk).strip() != ""]

    return clean_chunks


# def split_text_into_token_chunks(text, chunk_size, overlap):
#     tokens = st.session_statetokenizer.encode(text, add_special_tokens=False)
#     chunks = []
#     start = 0
#     while start < len(tokens):
#         end = start + chunk_size
#         chunk_tokens = tokens[start:end]
#         chunk_text = st.session_state.tokenizer.decode(chunk_tokens)
#         chunks.append(chunk_text)
#         start += chunk_size - overlap

#     return chunks


# def get_final_chunks(chunks, chunk_size, overlap):
#     final_chunks = []
#     for chunk in chunks:
#         chunk_tokens = st.session_state.tokenizer.encode(chunk, add_special_tokens=False)
#         if len(chunk_tokens) > chunk_size:
#             token_chunks = split_text_into_token_chunks(chunk, chunk_size=chunk_size, overlap=overlap)
#             final_chunks.extend(token_chunks)
#         else:
#             final_chunks.append(chunk)

#     return final_chunks


def get_content(retrieved_documents: List[Dict[str, str]]) -> str:
    if not retrieved_documents:
        return ""

    sections: List[str] = []
    for document in retrieved_documents:
        metadata = document.get("metadata", {}) or {}
        source = metadata.get("source", "Document")
        sections.append(f"--- {source} ---\n{document.get('content', '')}")

    return "\n\n".join(sections)


def extract_sources(retrieved_documents: List[Dict[str, str]]) -> List[str]:
    sources = []
    for document in retrieved_documents:
        metadata = document.get("metadata", {}) or {}
        source = metadata.get("source")
        if source and source not in sources:
            sources.append(source)
    return sources


def pass_submit_button():
    st.session_state.submit_button = True

def pause_submit_button():
    st.session_state.submit_button = False
    st.session_state.dlt_vec_store = False

def LLM_model_changed():
    st.session_state.LLM_model_changed = True

def delete_vec_store():
    st.session_state.dlt_vec_store = True
    st.session_state.submit_button = False
    try:
        st.session_state.vector_database.delete_vector_store()
    except:
        pass


def trigger_rerun() -> None:
    """Trigger a Streamlit rerun across supported versions."""
    rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun is not None:
        rerun()
    else:
        raise RuntimeError("Streamlit rerun functionality is not available in this version.")


CONFIG_PATH = os.getenv("APP_CONFIG_PATH", "./configs/config.yaml")
with open(CONFIG_PATH, "r") as f:
   CONFIG = yaml.safe_load(f)


chat_history = ChatHistory(os.getenv("CHAT_HISTORY_PATH", os.path.join(os.path.dirname(__file__), "..", "chat_history.sqlite")))


def get_initial_message() -> Dict[str, str]:
    return {
        "role": "assistant",
        "content": f"How can I help you as a {CONFIG['app']['Industry']} industry assistant?",
    }


response = requests.get(CONFIG['app']['logo'])
img = Image.open(BytesIO(response.content))
st.set_page_config(page_title=CONFIG['app']['title'],page_icon=img,layout="wide",initial_sidebar_state="expanded")


if "conversation_id" not in st.session_state:
    initial_message = get_initial_message()
    st.session_state.conversation_id = chat_history.create_conversation(
        title=None,
        initial_message=initial_message,
    )
    st.session_state.previous_messages = chat_history.load_conversation(st.session_state.conversation_id)

if "previous_messages" not in st.session_state:
    loaded_messages = chat_history.load_conversation(st.session_state.conversation_id)
    st.session_state.previous_messages = loaded_messages if loaded_messages else [get_initial_message()]

if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = st.session_state.conversation_id

if "LLM_client" not in st.session_state:
    st.session_state.LLM_client = None

if "LLM_model_changed" not in st.session_state:
    st.session_state.LLM_model_changed = True

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

if "vector_database" not in st.session_state:
    st.session_state.vector_database = None

if "previous_uploaded_file_names" not in st.session_state:
    st.session_state.previous_uploaded_file_names = []

if "submit_button" not in st.session_state:
    st.session_state.submit_button = False

if "dlt_vec_store" not in st.session_state:
    st.session_state.dlt_vec_store = False

st.session_state.setdefault("top_k_results", 5)

uploaded_files: List = []

with st.sidebar:
    st.logo(CONFIG["app"]["logo"], size="large")
    st.title(CONFIG['app']['title'])

    LLM_Model = st.selectbox("Select LLM Model", CONFIG["LLM_Models"]["supported"], index=0, key="LLM_model_key", on_change=LLM_model_changed)

    if st.session_state.LLM_model_changed:
        st.session_state.LLM_client = LLM_Assistant(industry=CONFIG['app']['Industry'], provider=LLM_Model)
        st.session_state.LLM_model_changed = False

    response_lang = st.selectbox("Select a response language", CONFIG["languages"]["supported"], index=0, accept_new_options=True, key="response_lang_key")

    st.markdown("## Import and chat with you PDF/s")
    uploaded_files = st.file_uploader("Upload PDF/s", accept_multiple_files=True, type="pdf", key="pdf_uploader_key", on_change=pause_submit_button)

    if uploaded_files == []:
        st.session_state.previous_uploaded_file_names = []
        try:
            st.session_state.vector_database.delete_vector_store()
        except Exception:
            pass
        st.session_state.embedding_model = None
        st.session_state.vector_database = None

    elif uploaded_files:
        embed_func = st.selectbox("Select Embedding Model", CONFIG["Embedding_Models"]["supported"], index=0, key="embedding_model_key", on_change=delete_vec_store)

        embedding_models_obj = Embedding_models(model_name=embed_func)
        st.session_state.embedding_model = embedding_models_obj.get_embedding_model()

        col1, col2 = st.columns(2)
        chunk_size = col1.number_input("Chunk Size", min_value=100, max_value=8000, value=1000, key="chunk_size_key", on_change=delete_vec_store)
        chunk_overlap = col2.number_input("Chunk Overlap", min_value=10, max_value=800, value=100, key="chunk_overlap_key", on_change=delete_vec_store)

        current_file_names = [uploaded_file.name for uploaded_file in uploaded_files]
        vector_store_provider = st.selectbox("Select Vector Store", CONFIG["vector_store"]["supported"], index=0, key="vector_store_key", on_change=delete_vec_store)

        st.button("Submit", on_click=pass_submit_button, key="submit", type="primary", disabled=st.session_state.submit_button)

        top_k_results = st.slider(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=st.session_state.get("top_k_results", 5),
            step=1,
            key="top_k_results_key",
        )
        st.session_state.top_k_results = top_k_results

        if st.session_state.submit_button:
            previous_files = st.session_state.previous_uploaded_file_names
            needs_new_store = (
                st.session_state.vector_database is None
                or st.session_state.dlt_vec_store
                or not previous_files
            )

            if needs_new_store:
                st.session_state.vector_database = Vector_stores(provider=vector_store_provider, embedding_model=st.session_state.embedding_model)

            files_to_process = current_file_names if needs_new_store else [
                name for name in current_file_names if name not in previous_files
            ]

            for uploaded_file in uploaded_files:
                if uploaded_file.name in files_to_process:
                    md_texts = get_texts(uploaded_file)
                    chunks = get_chunks(md_texts, chunk_size, chunk_overlap)
                    embeddings = st.session_state.embedding_model.encode(chunks)
                    metadatas = [{"source": uploaded_file.name}] * len(chunks)
                    ids = [f"{uploaded_file.name}_{i}" for i in range(len(chunks))]

                    st.session_state.vector_database.add_embeddings(ids=ids, texts=chunks, embeddings=embeddings, metadatas=metadatas)

            if st.session_state.vector_database is not None and len(current_file_names) < len(previous_files):
                pdf_names_to_remove = list(set(previous_files) - set(current_file_names))
                for pdf_name in pdf_names_to_remove:
                    try:
                        st.session_state.vector_database.delete_embeddings(filter={"source": pdf_name})
                    except AttributeError:
                        pass

            st.session_state.previous_uploaded_file_names = current_file_names
            st.session_state.dlt_vec_store = False

    st.markdown("---")
    st.markdown("## Conversation history")

    conversations = chat_history.list_conversations()
    conversation_options = [conversation.id for conversation in conversations]

    if conversation_options:
        try:
            default_index = conversation_options.index(st.session_state.conversation_id)
        except ValueError:
            default_index = 0

        def format_conversation(option_id: str) -> str:
            for conversation in conversations:
                if conversation.id == option_id:
                    timestamp = conversation.created_at.split("T")[0]
                    return f"{conversation.title} ({timestamp})"
            return option_id

        selected_conversation = st.selectbox(
            "View past sessions",
            options=conversation_options,
            index=default_index,
            format_func=format_conversation,
            key="conversation_history_select",
        )

        if selected_conversation != st.session_state.active_conversation:
            st.session_state.conversation_id = selected_conversation
            st.session_state.previous_messages = chat_history.load_conversation(selected_conversation) or [get_initial_message()]
            st.session_state.active_conversation = selected_conversation
            trigger_rerun()

    if st.button("Start new conversation", use_container_width=True):
        new_conversation_id = chat_history.create_conversation(initial_message=get_initial_message())
        st.session_state.conversation_id = new_conversation_id
        st.session_state.previous_messages = chat_history.load_conversation(new_conversation_id)
        st.session_state.active_conversation = new_conversation_id
        trigger_rerun()

for message in st.session_state.previous_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("What is up?"):
    st.session_state.previous_messages.append({"role": "user", "content": query})
    chat_history.save_message(
        st.session_state.conversation_id,
        role="user",
        content=query,
    )
    with st.chat_message("user"):
        st.markdown(query)

    retrieved_documents: List[Dict[str, str]] = []
    content = ""
    retrieved_sources: List[str] = []
    top_k_results = st.session_state.get("top_k_results", 5)

    try:
        if st.session_state.embedding_model is not None and st.session_state.vector_database is not None:
            query_embeddings = st.session_state.embedding_model.encode(query)
            retrieved_documents = st.session_state.vector_database.retrieve_documents(
                query_embeddings=query_embeddings,
                query=query,
                n_results=top_k_results,
            ) or []
            content = get_content(retrieved_documents)
            retrieved_sources = extract_sources(retrieved_documents)
    except AttributeError:
        retrieved_documents = []
        content = ""

    doc_titles = list(st.session_state.previous_uploaded_file_names)
    for source in retrieved_sources:
        if source not in doc_titles:
            doc_titles.append(source)

    assistant_reply = ""
    with st.chat_message("assistant"):
        if st.session_state.LLM_client is None:
            st.warning("LLM client is not configured. Please select a model to continue.")
        else:
            try:
                response_stream = st.session_state.LLM_client.get_response(
                    query,
                    doc_titles=doc_titles,
                    retrieved_content=content,
                    response_lang=response_lang,
                )

                if isinstance(response_stream, str):
                    st.markdown(response_stream)
                    assistant_reply = response_stream
                else:
                    response_output = st.write_stream(response_stream)
                    assistant_reply = response_output or st.session_state.LLM_client.last_response or ""
            except Exception as error:
                st.error(f"Failed to generate response: {error}")
                assistant_reply = "I encountered an error while generating a response."

    assistant_reply = assistant_reply.strip()
    if not assistant_reply:
        if content:
            assistant_reply = content
        else:
            assistant_reply = "I don't have enough information to answer this question."

    st.session_state.previous_messages.append({"role": "assistant", "content": assistant_reply})
    chat_history.save_message(
        st.session_state.conversation_id,
        role="assistant",
        content=assistant_reply,
    )