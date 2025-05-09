from langchain_chroma import Chroma
from retriever import Retriever
from langchain_ollama import OllamaEmbeddings
import re
import ollama
from memory import create_summary_memory


class MDUBot:
    def __init__(self, model_name="gemma3:4b", embed_model_name="mxbai-embed-large", persist_path="./chroma"):
        self.model = model_name
        self.embed_model = OllamaEmbeddings(model=embed_model_name)

        self.db = Chroma(embedding_function=self.embed_model, persist_directory=persist_path)
        self.retriever = Retriever(self.db, self.embed_model)
        self.memory = create_summary_memory(model_name=self.model)

    def run(self):
        test_prompts = [
            "What are the prerequisites for DVA494?",
            "What are the prerequisites for CDT406?",
            "Is that course active this year?",
            "exit"
        ]

        while True:
            prompt = test_prompts.pop(0)
            print(f"\nTest prompt: {prompt}")
            if prompt.lower().strip() == "exit":
                break

            prompt_cleaned = prompt.lower().strip()
            course_code = re.findall(r'\b[a-z]{2,3}\d{3}\b', prompt_cleaned)
            program_code = re.findall(r'\b[a-z]{2,3}\d{2}\b', prompt_cleaned)
            num_codes = len(course_code) or len(program_code)

            context = self.retriever.query(prompt_cleaned, course_code, program_code, num_codes)
            context_text = "\n".join([doc.page_content for doc in context]) if context else "No documents found."

            summary_context = self.memory.load_memory_variables({})["chat_history"]

            system_prompt = f"""You are an assistant helping answer questions about university courses and programs at Mälardalens universitet (MDU).

Here is the summary of the conversation so far:
{summary_context}

Here is the new context:
{context_text}

This is the current question: {prompt}

Answer the question by:
- Providing relevant information from the context.
- Using your knowledge to generate a response.
- Ensuring the response is accurate and helpful.
- Using the correct course or program codes when referring to specific courses or programs.
- Referring to the university as Mälardalens universitet or MDU. Do not use MDH or Mälardalens Högskola.
- Answer in the same language as the question.
"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": system_prompt}]
            )
            bot_reply = response['message']['content']

            print(f"\nMDUBot: {bot_reply}")

            self.memory.chat_memory.add_user_message(prompt)
            self.memory.chat_memory.add_ai_message(bot_reply)


if __name__ == "__main__":
    bot = MDUBot()
    bot.run()
