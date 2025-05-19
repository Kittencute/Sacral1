from langchain_chroma import Chroma
from testretriever import Retriever
from langchain_ollama import OllamaEmbeddings
import re
import ollama
from eval_tester import EvalTester
from testprompts import test_prompts

cosine_scores = []
bert_scores = []


class MDUBot:
    def __init__(self, model_name="gemma3-4b-ctx20k", embed_model_name="mxbai-embed-large", persist_path="./chroma"):
        # Start the model, embeddings and connect to chroma database
        self.model = model_name
        self.embed_model = OllamaEmbeddings(model=embed_model_name)
        self.db = Chroma(embedding_function=self.embed_model, persist_directory=persist_path)
        self.retriver = Retriever(self.db, self.embed_model)

        self.evaluator = EvalTester(self.embed_model)

    def preprocess_query(self, prompt):
        # Try to find course or program codes using regex
        course_code = re.findall(r'\b[a-z]{2,3}\d{3}\b', prompt.lower())
        program_code = re.findall(r'\b[a-z]{2,3}\d{2}\b', prompt.lower())
        topic_keywords = []

        # Get metadata from the database
        all_metadata = self.db._collection.get(include=["metadatas"])["metadatas"]

        # Collect all course and program codes from metadata
        all_course_codes = {meta.get('course_code', '').lower() for meta in all_metadata if 'course_code' in meta}
        all_program_codes = {meta.get('program_code', '').lower() for meta in all_metadata if 'program_code' in meta}
        course_name_to_code = {
            meta.get('course_name', '').lower(): meta.get('course_code', '')
            for meta in all_metadata if 'course_name' in meta and 'course_code' in meta
        }

        # Keep only valid codes that are actually in the database
        valid_course_code = [code for code in course_code if code in all_course_codes]
        valid_program_code = [code for code in program_code if code in all_program_codes]

        # Extract course names from prompt
        found_course_names = []
        prompt_lower = prompt.lower()
        for cname in course_name_to_code:
            if cname and cname in prompt_lower and cname not in found_course_names:
                found_course_names.append(cname)

        # Map found names to codes if not already in valid_course_code
        for name in found_course_names:
            code = course_name_to_code.get(name)
            if code and code.lower() not in valid_course_code:
                valid_course_code.append(code.lower())

        # If no codes found, classify and extract topic keywords
        if not valid_course_code and not valid_program_code:
            classification = ollama.chat(
                model="gemma3-4b-ctx20k",
                messages=[
                    {"role": "system", "content": "Classify the user question. Answer only with one word: Course, Program, Topic."},
                    {"role": "user", "content": prompt}
                ]
            )["message"]["content"].lower().strip()

            # If its a topic, ask for keywords
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
        """
        Retrieve and prepare metadata mappings for course codes and names.
        """
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

    def run(self):
        # Show welcome message with example questions
        print("""
Welcome! I can help you explore university courses and programs.

Try asking things like:

• What is [course code / course name] or [program code / program name] about?
  eg: What is CDT406? Tell me about Tillämpad artificiell intelligens?

• Can I replace [course] in [program]?
  eg: Can I replace Mekatronik in Civilingenjörsprogrammet i robotik?

• Recommend courses or programs related to [topic]
  eg: Recommend advanced math courses.
""")
        
        while True:
            item = test_prompts.pop(0)
            prompt = item["prompt"]
            reference = item["reference"]
            print(f"\nTest prompt: {prompt}")

            if prompt == "exit":
                if cosine_scores and bert_scores:
                    avg_cosine = sum(cosine_scores) / len(cosine_scores)
                    avg_bert = sum(bert_scores) / len(bert_scores)
                    self.evaluator.log_average_scores(avg_cosine, avg_bert)
                break

            # Preprocess the query to extract course code, program code, and topic keywords
            course_code, program_code, found_course_names, topic_keywords = self.preprocess_query(prompt)

            print(f"Course code: {course_code}")
            print(f"Program code: {program_code}")
            print(f"Course names: {found_course_names}")
            print(f"Topic keywords: {topic_keywords}")
            print(f"Retrieving for: {' '.join(topic_keywords) if topic_keywords else prompt}")

            # --- Initialize metadata mappings ---
            all_metadata, course_name_to_code, code_to_name = self.get_metadata_mappings()
            context_sections = []
            used_codes = set()

            # --- Retrieve documents for course or program codes ---
            if course_code or program_code:
                all_course_codes = course_code + [course_name_to_code.get(cname) for cname in found_course_names if course_name_to_code.get(cname)]
                docs = self.retriver.query_multiple(prompt, course_codes=all_course_codes, num_codes=5)
                if docs:
                    context = "\n".join([doc.page_content for doc in docs])
                    context_sections.append(f"=== Retrieved Documents ===\n{context}\n")
            elif topic_keywords:
                # Use all extracted topic keywords as a single prompt
                topic_query = " ".join(topic_keywords)
                docs = self.retriver.query(topic_query, num_codes=15)
                if docs:
                    context = "\n".join([doc.page_content for doc in docs])
                    context_sections.append(f"=== Retrieved Topic Documents ===\n{context}\n")
            else:
                # --- Fallback to general search if no specific context found ---
                docs = self.retriver.query(prompt, num_codes=15)
                if docs:
                    context = "\n".join([doc.page_content for doc in docs])
                    context_sections.append(f"=== General context ===\n{context}\n")

            result = "\n".join(context_sections)

            # Create a new prompt to send to the LLM with the context
            full_prompt = f"""You are an assistant helping answer questions about university courses and programs at Mälardalens universitet (MDU).
Here is the context about the course(s) or program(s):
{result}
This is the question: {prompt}

Answer the question by:
- Providing relevant information from the context, clearly separated for each course or program if multiple are mentioned.
- Using your knowledge to generate a response.
- Ensuring the response is accurate and helpful.
- Using the correct course or program codes when referring to specific courses or programs.
- Referring to the university as Mälardalens universitet or MDU. Do not use MDH or Mälardalens Högskola, as these are old abbreviations.
- Answer in the same language as the question provided.
"""

            # Send the prompt to the LLM and get the response
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            # Ensure proper Unicode handling for the LLM response
            try:
                final_response = response["message"]["content"].encode('utf-8').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                final_response = response["message"]["content"]

            print(f"\nMDUBot: {final_response}\n")

            cos_score = self.evaluator.compute_cosine_similarity(prompt, final_response)
            cosine_scores.append(cos_score)
            if reference.strip():
                bert_score = self.evaluator.compute_bertscore(final_response, reference)
                bert_scores.append(bert_score)
            else:
                bert_score = None

            self.evaluator.log_evaluation(prompt, final_response, cos_score, bert_score, reference)


if __name__ == "__main__":
    bot = MDUBot()
    bot.run()

