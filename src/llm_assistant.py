import os
import time
from mistralai import Mistral
from openai import OpenAI

class LLM_Assistant:
    def __init__(self, industry, provider):
        self.industry = industry
        self.provider = provider
        self.last_response = None

    def _prompt_template(self, query, content, response_lang):
        system_msg = {
            "role": "system",
            "content": f"""You are a helpful AI {self.industry} assistant. You must only answer questions using the information found in the provided content. You must strictly 
            follow the rules below when responding to the user's question.
            
            Rules:-
            1. If the user's question is about the {self.industry} Industry → 
                - If the answer is in the PDF → give the answer.
                - If the answer is NOT in the PDF → reply exactly: "I don't have enough information to answer this question."

            2. If the user's question is NOT about the {self.industry} Industry →  
                - If the question is about yourself or how you work or non industry questions → answer it normally.
                - Otherwise, reply exactly: "Sorry, I'm a {self.industry} AI assistant. I can't help with this question. Please ask questions from the {self.industry} industry."

            3. Always provide the final answer only in {response_lang} Language."""
        }

        user_msg = {
            "role": "user",
            "content": f"""=== Question ===\n\n{query}\n\n\n=== Content ===\n\n{content}"""
        }

        return [system_msg, user_msg]


    # def stream(self, stream):
    #     if self.provider == "Mistral/mistral-large-latest":
    #         full_response = ""
    #         for chunk in stream:
    #             response = chunk.data.choices[0].delta.content
    #             yield response
    #             full_response += response
    #             time.sleep(0.02)
    #         self.last_response = "".join(full_response)
            
        
    #     elif self.provider == "OpenAI/gpt-oss-120b":
    #         full_response = ""
    #         for chunk in stream:
    #             try:
    #                 response = chunk.choices[0].delta.content
    #                 yield response
    #                 full_response += response
    #                 time.sleep(0.02)
    #             except:
    #                 pass
    #         self.last_response = "".join(full_response)

    def _streamed_response(self, stream_response):
        if self.provider == "Mistral/mistral-large-latest":
            full_response = []
            for chunk in stream_response:
                response = chunk.data.choices[0].delta.content
                if response:
                    yield response
                    full_response.append(response)
                    time.sleep(0.02)
            self.last_response = "".join(full_response)

        elif self.provider == "OpenAI/gpt-oss-120b":
            full_response = []
            for chunk in stream_response:
                try:
                    response = chunk.choices[0].delta.content
                    if response:
                        yield response
                        full_response.append(response)
                        time.sleep(0.02)
                except Exception:
                    pass
            self.last_response = "".join(full_response)
        
        elif self.provider == "DeepSeek/DeepSeek-R1":
            full_response = []
            after_think = True
            
            for chunk in stream_response:
                response = chunk.choices[0].delta.content
                if response:
                    if after_think:
                        if "</think>" in response:
                            after_think = False
                            response = response.split("</think>", 1)[-1].strip()
                        else:
                            continue

                    if response:
                        yield response
                        full_response.append(response)
                            
                    time.sleep(0.02)
            
            self.last_response = "".join(full_response)

        elif self.provider == "Meta/Llama-3.1-8B-Instruct":
            full_response = []
            for chunk in stream_response:
                try:
                    response = chunk.choices[0].delta.content
                    if response:
                        yield response
                        full_response.append(response)
                        time.sleep(0.02)
                except Exception:
                    pass

            self.last_response = "".join(full_response)

    def get_response(self, query, content, response_lang):
        messages = self._prompt_template(query, content, response_lang)

        if self.provider == "Mistral/mistral-large-latest":
            client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
            response_stream = client.chat.stream(
                model="mistral-large-latest",
                messages=messages)
            return response_stream

        elif self.provider == "OpenAI/gpt-oss-120b":
            client = OpenAI(base_url=os.getenv("ROUTER_BASE_URL", "https:https://router.huggingface.co/v1"), api_key=os.getenv("OPENAI_API_KEY"))
            response_stream = client.chat.completions.create(
                model="openai/gpt-oss-120b:together",
                messages=messages,
                stream=True)
            return self._streamed_response(response_stream)
        
        elif self.provider == "DeepSeek/DeepSeek-R1":
            client = OpenAI(base_url=os.getenv("ROUTER_BASE_URL", "https:https://router.huggingface.co/v1"), api_key=os.getenv("DEEPSEEK_API_KEY"))

            response_stream = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1:together",
                messages=messages,
                stream=True)
            return self._streamed_response(response_stream)
        
        elif self.provider == "Meta/Llama-3.1-8B-Instruct":
            client = OpenAI(base_url=os.getenv("ROUTER_BASE_URL", "https:https://router.huggingface.co/v1"), api_key=os.getenv("META_API_KEY"))

            response_stream = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct:cerebras",
                messages=messages,
                stream=True)

            return self._streamed_response(response_stream)

        else:
            raise ValueError("Unsupported provider. Use 'openai' or 'mistral'.")
        
