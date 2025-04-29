from langchain_chroma import Chroma
from retriever import Retriever
from langchain_ollama import OllamaEmbeddings
import re
import ollama


class MDUBot:
    def __init__(self, model_name="gemma3:4b", embed_model_name="mxbai-embed-large", persist_path="./chroma"):
        # Start the model, embeddings and connect to chroma database
        self.model = model_name
        self.embed_model = OllamaEmbeddings(model=embed_model_name)
        self.db = Chroma(embedding_function=self.embed_model, persist_directory=persist_path)
        self.retriver = Retriever(self.db, self.embed_model)

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

        # Keep only valid codes that are actually in the database
        valid_course_code = [code for code in course_code if code in all_course_codes]
        valid_program_code = [code for code in program_code if code in all_program_codes]

        # If no codes found, classify and extract topic keywords
        if not valid_course_code and not valid_program_code:
            classification = ollama.chat(
                model="gemma3:4b",
                messages=[
                    {"role": "system", "content": "Classify the user question. Answer only with one word: Course, Program, Topic."},
                    {"role": "user", "content": prompt}
                ]
            )["message"]["content"].lower().strip()

            # If its a topic, ask for keywords
            if classification == "topic":
                topic_keywords = ollama.chat(
                    model="gemma3:4b",
                    messages=[
                        {"role": "system", "content": "Extract important topic keywords from this question. Only list important words separated by commas. No explanation."},
                        {"role": "user", "content": prompt}
                    ]
                )["message"]["content"].lower().strip().split(",")

                topic_keywords = [k.strip() for k in topic_keywords if k.strip()]

        return valid_course_code, valid_program_code, topic_keywords

    def run(self):
        test_prompts = [
            "What are the examination components in CDT406?",
            "What are the prerequisites for CDT406?",
            "Can you summarize what CDT406 is about?",
            "I like mathematics, can you recommend specific courses related to this?",
            "Which courses are included in the CCV20 program?",
            "exit"
        ]

        while True:
            prompt = test_prompts.pop(0)
            print(f"\nTest prompt: {prompt}")

            if prompt == "exit":
                break

            # Preprocess the query to extract course code, program code, and topic keywords
            course_code, program_code, topic_keywords = self.preprocess_query(prompt)

            print(f"Course code: {course_code}")
            print(f"Program code: {program_code}")
            print(f"Topic keywords: {topic_keywords}")
            print(f"Retrieving for: {' '.join(topic_keywords) if topic_keywords else prompt}")

            # Decide how to search based on the extracted codes
            if course_code:
                selected_code = course_code
                selected_program = None
                num_codes = len(course_code)
                prompt_for_search = course_code[0]  # Search with course code
            elif program_code:
                selected_code = None
                selected_program = program_code
                num_codes = len(program_code)
                prompt_for_search = program_code[0]  # Search with program code
            else:
                selected_code = None
                selected_program = None
                num_codes = 5
                prompt_for_search = " ".join(topic_keywords) if topic_keywords else prompt

            # Ask the retriever for relevant documents
            result_docs = self.retriver.query(
                prompt_for_search,
                course_code=selected_code,
                program_code=selected_program,
                num_codes=num_codes
            )
            result = "\n".join([doc.page_content for doc in result_docs])

            # Create a new promt to send to the LLM with the context
            full_prompt = f"""You are an assistant helping answer questions about university courses and programs at Mälardalens universitet (MDU).
Here is the context about the course or program:\n{result}\n
This is the question: {prompt}\n
Answer the question by:
- Providing relevant information from the context.
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

            print(f"\nMDUBot: {response['message']['content']}\n")


if __name__ == "__main__":
    bot = MDUBot()
    bot.run()

