from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from testretriever import Retriever
from langchain_ollama import OllamaEmbeddings
import re
import ollama
from eval_tester import EvalTester

app = FastAPI()

# Allow CORS for local development and web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class MDUBot:
    def __init__(self, model_name="gemma3-4b-ctx20k", embed_model_name="mxbai-embed-large", persist_path="./chroma"):
        self.model = model_name
        self.embed_model = OllamaEmbeddings(model=embed_model_name)
        self.db = Chroma(embedding_function=self.embed_model, persist_directory=persist_path)
        self.retriver = Retriever(self.db, self.embed_model)
        self.evaluator = EvalTester(self.embed_model)

    def preprocess_query(self, prompt):
        course_code = re.findall(r'\b[a-z]{2,3}\d{3}\b', prompt.lower())
        program_code = re.findall(r'\b[a-z]{2,3}\d{2}\b', prompt.lower())
        topic_keywords = []

        all_metadata = self.db._collection.get(include=["metadatas"])["metadatas"]

        all_course_codes = {meta.get('course_code', '').lower() for meta in all_metadata if 'course_code' in meta}
        all_program_codes = {meta.get('program_code', '').lower() for meta in all_metadata if 'program_code' in meta}
        course_name_to_code = {
            meta.get('course_name', '').lower(): meta.get('course_code', '')
            for meta in all_metadata if 'course_name' in meta and 'course_code' in meta
        }

        valid_course_code = [code for code in course_code if code in all_course_codes]
        valid_program_code = [code for code in program_code if code in all_program_codes]

        found_course_names = []
        prompt_lower = prompt.lower()
        for cname in course_name_to_code:
            if cname and cname in prompt_lower and cname not in found_course_names:
                found_course_names.append(cname)

        for name in found_course_names:
            code = course_name_to_code.get(name)
            if code and code.lower() not in valid_course_code:
                valid_course_code.append(code.lower())

        if not valid_course_code and not valid_program_code:
            classification = ollama.chat(
                model="gemma3-4b-ctx20k",
                messages=[
                    {"role": "system", "content": "Classify the user question. Answer only with one word: Course, Program, Topic."},
                    {"role": "user", "content": prompt}
                ]
            )["message"]["content"].lower().strip()

            topic_keywords_raw = ollama.chat(
                model="gemma3-4b-ctx20k",
                messages=[
                    {"role": "system", "content": "Extract important topic keywords from this question. Only list important words separated by commas. No explanation."},
                    {"role": "user", "content": prompt}
                ]
            )["message"]["content"].lower().strip()

            topic_keywords = [k.strip() for k in topic_keywords_raw.split(",") if k.strip()]

        return valid_course_code, valid_program_code, found_course_names, topic_keywords

    def get_metadata_mappings(self):
        all_metadata = self.db._collection.get(include=["metadatas"])["metadatas"]
        course_name_to_code = {
            meta.get('course_name', '').lower(): meta.get('course_code', '')
            for meta in all_metadata if 'course_name' in meta and 'course_code' in meta
        }
        code_to_name = {
            meta.get('course_code', '').lower(): meta.get('course_name', '')
            for meta in all_metadata if 'course_code' in meta
        }
        return all_metadata, course_name_to_code, code_to_name

    def single_turn_chat(self, prompt):
        course_code, program_code, found_course_names, topic_keywords = self.preprocess_query(prompt)
        all_metadata, course_name_to_code, code_to_name = self.get_metadata_mappings()
        context_sections = []
        if course_code or program_code:
            all_course_codes = course_code + [course_name_to_code.get(cname) for cname in found_course_names if course_name_to_code.get(cname)]
            docs = self.retriver.query_multiple(prompt, course_codes=all_course_codes, num_codes=5)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                context_sections.append(f"=== Retrieved Documents ===\n{context}\n")
        elif topic_keywords:
            math_keywords = [kw for kw in topic_keywords if "math" in kw or "algebra" in kw or "calculus" in kw]
            ai_keywords = [kw for kw in topic_keywords if "ai" in kw or "artificial" in kw or "intelligence" in kw]
            if math_keywords:
                math_query = " ".join(math_keywords)
                math_docs = self.retriver.query(math_query, num_codes=10)
                if math_docs:
                    math_context = "\n".join([doc.page_content for doc in math_docs])
                    context_sections.append(f"=== Retrieved Math Documents ===\n{math_context}\n")
            if ai_keywords:
                ai_query = " ".join(ai_keywords)
                ai_docs = self.retriver.query(ai_query, num_codes=10)
                if ai_docs:
                    ai_context = "\n".join([doc.page_content for doc in ai_docs])
                    context_sections.append(f"=== Retrieved AI Documents ===\n{ai_context}\n")
        else:
            docs = self.retriver.query(prompt, num_codes=15)
            if docs:
                context = "\n".join([doc.page_content for doc in docs])
                context_sections.append(f"=== General context ===\n{context}\n")
        result = "\n".join(context_sections)
        full_prompt = f"""You are an assistant helping answer questions about university courses and programs at Mälardalens universitet (MDU).
Here is the context about the course(s) or program(s):
{result}
This is the question: {prompt}

Please answer the question by:
- Structuring your answer clearly, using headings, bullet points, and lists where appropriate.
- Use Markdown formatting for headings, bold, and lists.
- Separate different courses or programs with clear headings.
- Keep answers concise and easy to scan.
- Use the correct course or program codes when referring to specific courses or programs.
- Refer to the university as Mälardalens universitet or MDU. Do not use MDH or Mälardalens Högskola, as these are old abbreviations.
- Answer in the same language as the question provided.
"""
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        try:
            final_response = response["message"]["content"].encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            final_response = response["message"]["content"]
        return final_response

# Instantiate the bot once for the API
bot = MDUBot()

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    response = bot.single_turn_chat(req.message)
    return {"response": response}