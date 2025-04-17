import chroma_patch

from langchain_chroma import Chroma
from retriever import Retriever
from langchain_ollama import OllamaEmbeddings
import re
import ollama


class MDUBot:
    # Change model_name, embed_model_name or persist_path if needed
    # model_name = "llama3.1:8b" or "deepseek-r1:7b"
    def __init__(self, model_name="gemma3:4b", embed_model_name="mxbai-embed-large", persist_path="./chroma"): # Change persist_path if needed
        self.model = model_name
        self.embed_model = OllamaEmbeddings(model=embed_model_name)
        self.db = Chroma(embedding_function=self.embed_model, persist_directory=persist_path)
        self.retriver = Retriever(self.db, self.embed_model)

    def run(self):
        while True:
            prompt = input("You: ")
            if prompt == "exit":
                break

            prompt = prompt.lower().strip()
            course_code = re.findall(r'\b[a-z]{2,3}\d{3}\b', prompt)
            num_codes = len(course_code)
            program_code = re.findall(r'\b[a-z]{2,3}\d{2}\b', prompt)
            num_codes = len(program_code) if num_codes == 0 else num_codes
            program_code = program_code

            result = self.retriver.query(prompt, course_code, program_code, num_codes)

            prompt = f"""You are an assistant helping answer questions about university courses and programs at Mälardalens universitet (MDU).
                        Here is the context about the course or program:\n{result}\n
                        This is the question: {prompt}\n
                        Answer the question by:
                        Providing relevant information from the context.
                        Using your knowledge to generate a response.
                        Ensuring the response is accurate and helpful.
                        Using the correct course or program codes when referring to specific courses or programs.
                        Referring to the university as Mälardalens universitet or MDU. Do not use MDH or Mälardalens Högskola, as these are old abbreviations.
                        Answer in the same language as the question provided.
                        """

            # response = ollama.generate(model=self.model, prompt=prompt)
            # print(f"\nMDUBot: {response["response"]}")
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            print(f"\nMDUBot: {response['message']['content']}")


bot = MDUBot()
bot.run()