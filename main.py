from langchain_chroma import Chroma
from retriever import Retriever
from langchain_ollama import OllamaEmbeddings
import re
import ollama


class MDUBot:
    def __init__(self, model_name="gemma3:4b", embed_model_name="mxbai-embed-large", persist_path="./chroma"): # Change persist_path if needed
        self.model = model_name
        self.embed_model = OllamaEmbeddings(model=embed_model_name)
        self.db = Chroma(embedding_function=self.embed_model, persist_directory=persist_path)
        self.retriver = Retriever(self.db, self.embed_model)

    def run(self):

        test_prompts = [
            "What are the examination components in CDT406?",
            "What are the prerequisites for CDT406?",
            "Can you summarize what CDT406 is about?",
            "I like mathematics, can you recommend specific courses related to this?",
            "Which courses are included in the CCV20 program?",
            "exit"  
        ]
        # test_prompts = [
        #     "What are the prerequisites for CDT406?",
        #     "exit"  
        # ]
        
        while True:
            prompt = test_prompts.pop(0) 
            print(f"Test prompt: {prompt}")
            # prompt = input("You: ")
            if prompt == "exit":
                break

            preprocessed_query = ollama.chat(
                    model="gemma3:4b",
                messages=[
                    {"role": "system", "content": "Rephrase the following user question into a more specific and complete version suitable for                  retrieving relevant university course or program information from Mälardalens univeritet. Summerize the question into a compact form, format the answer into single important words and dont answer with any extra text or eplanation Answer in this format, Intent: , Coursecode: , Coursename:. If the intent, coursecode or coursename is not found, answer with NONE"},
                    {"role": "user", "content": prompt}
                ]
            )["message"]["content"]
            
            

            print("Preproccessed: " + preprocessed_query)

            preprocessed_query = preprocessed_query.lower().strip()
            course_code = re.findall(r'\b[a-z]{2,3}\d{3}\b', preprocessed_query)
            num_codes = len(course_code)
            program_code = re.findall(r'\b[a-z]{2,3}\d{2}\b', preprocessed_query)
            num_codes = len(program_code) if num_codes == 0 else num_codes
            program_code = program_code


            # preprocessed_query = ollama.chat(
            #         model="gemma3:4b",
            #     messages=[
            #         {"role": "system", "content": "Rephrase the following user question into a more specific and complete version suitable for                  retrieving relevant university course or program information from Mälardalens univeritet. Answer in this format, Intent: , Coursecode: , Coursename:. If the intent, coursecode or coursename is not found, answer with NONE"},
            #         {"role": "user", "content": prompt}
            #     ]
            # )["message"]["content"]

            result = self.retriver.query(preprocessed_query, course_code, program_code, num_codes)
            # result = self.retriver.query(prompt, course_code, program_code, num_codes)

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

            print(f"\nMDUBot: {response['message']['content']}\n")


bot = MDUBot()
bot.run()

