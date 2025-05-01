# Remove this if needed 
import chroma_patch

from langchain_chroma import Chroma
from retriever import Retriever
from langchain_ollama import OllamaEmbeddings
import re
import ollama
from eval_tester import EvalTester
from test_data import test_prompts

cosine_scores = []
bert_scores = []
class MDUBot:
    # Change model_name, embed_model_name or persist_path if needed
    # model_name = "gemma3:4b" or :1b
    def __init__(self, model_name="gemma3:4b", embed_model_name="mxbai-embed-large", persist_path="./chroma"): 
        self.model = model_name
        self.embed_model = OllamaEmbeddings(model=embed_model_name)
        self.db = Chroma(embedding_function=self.embed_model, persist_directory=persist_path)
        self.retriver = Retriever(self.db, self.embed_model)

        self.evaluator = EvalTester(self.embed_model)
        
    def run(self):        
        while True:
            item = test_prompts.pop(0) 
            user_prompt = item["prompt"]
            reference = item["reference"]
            print(f"Test prompt: {user_prompt}")
            # prompt = input("You: ")
            if user_prompt == "exit":
                if cosine_scores and bert_scores:
                    avg_cosine = sum(cosine_scores) / len(cosine_scores)
                    avg_bert = sum(bert_scores) / len(bert_scores)
                    self.evaluator.log_average_scores(avg_cosine, avg_bert)
                            
                break

            user_prompt = user_prompt.lower().strip()
            course_code = re.findall(r'\b[a-z]{2,3}\d{3}\b', user_prompt)
            num_codes = len(course_code)
            program_code = re.findall(r'\b[a-z]{2,3}\d{2}\b', user_prompt)
            num_codes = len(program_code) if num_codes == 0 else num_codes
            program_code = program_code
            
            result = self.retriver.query(user_prompt, course_code, program_code, num_codes)
            
            llm_prompt = f"""You are an assistant helping answer questions about university courses and programs at Mälardalens universitet (MDU).
                        Here is the context about the course or program:\n{result}\n
                        This is the question: {user_prompt}\n
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
                    {"role": "user", "content": llm_prompt}
                ]
            )
            final_response = response['message']['content']
            print(f"\nMDUBot: {final_response}")

            cos_score = self.evaluator.compute_cosine_similarity(user_prompt, final_response)
            cosine_scores.append(cos_score)
            if reference.strip():
                bert_score = self.evaluator.compute_bertscore(final_response, reference)
                bert_scores.append(bert_score)
            else:
                bert_score = None 
    
            self.evaluator.log_evaluation(user_prompt, final_response, cos_score, bert_score, reference)
            
bot = MDUBot()
bot.run()