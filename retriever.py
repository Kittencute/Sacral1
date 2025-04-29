from langchain_ollama import OllamaEmbeddings

class Retriever:
    def __init__(self, chroma_client, embed_model):
        self.embed_model = embed_model
        self.db = chroma_client

    def query(self, prompt, course_code=None, program_code=None, num_codes=1):
        docs = []

        # If course codes are found, search for each one and collect results
        if course_code:
            for code in course_code:
                results = self.db.similarity_search(prompt, k=num_codes, filter={"course_code": code})
                docs.extend(results)

        # If no course codes, try using program codes instead
        elif program_code:
            for code in program_code:
                results = self.db.similarity_search(prompt, k=num_codes, filter={"program_code": code})
                docs.extend(results)

        # If no course or program codes are found, perform a general search
        if not docs:
            docs = self.db.similarity_search(prompt, k=num_codes)

        # Try to fix any weird unicode in the results so they look better
        for doc in docs:
            try:
                doc.page_content = doc.page_content.encode().decode('unicode_escape')
            except Exception:
                pass # Ignore if decode fails

        return docs

