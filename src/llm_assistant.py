import os
import tempfile
import time
from typing import Iterable, List, Optional

import pymupdf4llm
from mistralai import Mistral
from openai import OpenAI


DEFAULT_ROUTER_URL = "https://router.huggingface.co/v1"


class LLM_Assistant:
    def __init__(self, industry: str, provider: str) -> None:
        self.industry = industry
        self.provider = provider
        self.last_response: Optional[str] = None

    def get_texts(self, uploaded_file) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        try:
            markdown_text = pymupdf4llm.to_markdown(temp_file_path)
        finally:
            os.remove(temp_file_path)

        return markdown_text

    def _prompt_template(
        self,
        query: str,
        doc_titles: Optional[Iterable[str]] = None,
        retrieved_content: str = "",
        response_lang: str = "English",
    ) -> List[dict]:
        titles = list(doc_titles or [])
        doc_count = len(titles)
        titles_str = "\n".join(f"{idx + 1}. {title}" for idx, title in enumerate(titles)) or "No documents uploaded"
        content_str = retrieved_content.strip() or "No content available."

        system_msg = {
            "role": "system",
            "content": f"""You are a helpful AI assistant specialized in IATA (International Air Transport Association), the Aviation Industry, and related Dangerous Goods regulations (including batteries such as lithium, lithium-ion, and sodium-ion).

You must always use the provided uploaded document content to answer questions, summarize, create training material, or generate quizzes.
If the document content is empty, reply exactly: "I don't have enough information to answer this question."

Rules:

1. Document Metadata Questions
   - If the user asks how many documents are uploaded → always reply with:
     "There are {doc_count} document(s) uploaded:" followed by a numbered list of document titles.
   - If the user asks for titles → always return the full list of document filenames, one per line.
   - Never use placeholder terms like "Uploaded Document". Always respond with the true filenames.

2. IATA / Aviation / Dangerous Goods Q&A
   - If the user's question relates to IATA, the Aviation Industry, Dangerous Goods, or topics covered in the uploaded documents:
     • If the answer exists in the provided document(s) → respond with the exact relevant answer.
     • Always include the document name and page number(s) where the answer was found using the format: (Document Name — Page X).
     • If the answer is NOT in the document(s) → reply exactly: "I don't have enough information to answer this question."

3. Summarization
   - Provide structured summaries strictly using the document content.
   - Mention the document title(s) at the start of the summary.
   - Include the document name and page number(s) with each key point.

4. Training / Lessons
   - When creating training material, structure the response into lessons highlighting definitions, classifications, and rules.
   - Reference the relevant document title(s) and page number(s).

5. Quiz / Mock Test Generation
   - Generate conceptual, logical, or regulatory questions only.
   - Include the correct answer after each question along with the cited document name and page number(s).
   - Mix multiple choice, true/false, and short answer formats.

6. Out-of-Scope Handling
   - If the question is unrelated to IATA, Aviation, Dangerous Goods, or the uploaded documents:
     • If it is about the AI itself → answer normally.
     • Otherwise → reply exactly: "Sorry, I'm an IATA AI assistant. I can't help with this question. Please ask questions related to IATA, the Aviation Industry, Dangerous Goods, or the uploaded documents."

7. Language Enforcement
   - The final answer must always be written entirely in {response_lang}.
""",
        }

        user_msg = {
            "role": "user",
            "content": f"""=== Question ===

{query}

=== Uploaded Document Titles ===
{titles_str}

=== Uploaded Document Content ===
{content_str}""",
        }

        return [system_msg, user_msg]

    def _stream_mistral(self, stream_response) -> Iterable[str]:
        def generator():
            full_response: List[str] = []
            for chunk in stream_response:
                try:
                    choice = chunk.data.choices[0]
                    delta = getattr(choice.delta, "content", None)
                except (AttributeError, IndexError):
                    delta = None
                if not delta:
                    continue
                yield delta
                full_response.append(delta)
                time.sleep(0.02)
            self.last_response = "".join(full_response)

        return generator()

    def _stream_openai_like(self, stream_response, *, strip_think: bool = False) -> Iterable[str]:
        def generator():
            full_response: List[str] = []
            waiting_for_think_close = strip_think
            for chunk in stream_response:
                try:
                    delta = getattr(chunk.choices[0].delta, "content", None)
                except (AttributeError, IndexError):
                    delta = None
                if not delta:
                    continue

                if waiting_for_think_close:
                    if "</think>" in delta:
                        waiting_for_think_close = False
                        delta = delta.split("</think>", 1)[-1]
                        if not delta:
                            continue
                        delta = delta.lstrip()
                    else:
                        continue

                if delta:
                    yield delta
                    full_response.append(delta)
                    time.sleep(0.02)

            self.last_response = "".join(full_response)

        return generator()

    def _build_router_client(self, *preferred_env_keys: str) -> OpenAI:
        base_url = os.getenv("ROUTER_BASE_URL", DEFAULT_ROUTER_URL)
        for key_name in preferred_env_keys:
            api_key = os.getenv(key_name)
            if api_key:
                return OpenAI(base_url=base_url, api_key=api_key)

        fallback = (
            os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HF_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not fallback:
            raise ValueError("A router API key is required. Please set HUGGINGFACE_API_KEY.")

        return OpenAI(base_url=base_url, api_key=fallback)

    def get_response(
        self,
        query: str,
        *,
        doc_titles: Optional[Iterable[str]] = None,
        retrieved_content: str = "",
        response_lang: str = "English",
    ) -> Iterable[str]:
        messages = self._prompt_template(
            query,
            doc_titles=doc_titles,
            retrieved_content=retrieved_content,
            response_lang=response_lang,
        )

        self.last_response = None

        if self.provider == "Mistral/mistral-large-latest":
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY environment variable is required for Mistral support.")

            client = Mistral(api_key=api_key)
            response_stream = client.chat.stream(
                model="mistral-large-latest",
                messages=messages,
            )
            return self._stream_mistral(response_stream)

        if self.provider == "OpenAI/gpt-oss-120b":
            client = self._build_router_client("OPENAI_API_KEY")
            response_stream = client.chat.completions.create(
                model="openai/gpt-oss-120b:together",
                messages=messages,
                stream=True,
            )
            return self._stream_openai_like(response_stream)

        if self.provider == "DeepSeek/DeepSeek-R1":
            client = self._build_router_client("DEEPSEEK_API_KEY")
            response_stream = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1:together",
                messages=messages,
                stream=True,
            )
            return self._stream_openai_like(response_stream, strip_think=True)

        if self.provider == "Meta/Llama-3.1-8B-Instruct":
            client = self._build_router_client("META_API_KEY")
            response_stream = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
                messages=messages,
                stream=True,
            )
            return self._stream_openai_like(response_stream)

        raise ValueError("Unsupported provider. Use a configured provider from the sidebar.")
