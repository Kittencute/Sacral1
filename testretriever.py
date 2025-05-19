from langchain_ollama import OllamaEmbeddings

class Retriever:
    def __init__(self, chroma_client, embed_model):
        self.embed_model = embed_model
        self.db = chroma_client

    def query(self, prompt, course_code=None, program_code=None, num_codes=1):
        docs = []

        # If course codes are found, search for each one and collect results
        if course_code:
            for code in set(course_code):
                results = self.db.similarity_search(prompt, k=num_codes, filter={"course_code": code})
                docs.extend(results)

        # If program codes are found, search for each one and collect results
        if program_code:
            for code in set(program_code):
                results = self.db.similarity_search(prompt, k=num_codes, filter={"program_code": code})
                docs.extend(results)

        # If no course or program codes are found, perform a general keyword-based search
        if not docs:
            results = self.db.similarity_search(prompt, k=num_codes)
            docs.extend(results)

        # Ensure proper Unicode handling
        for doc in docs:
            try:
                doc.page_content = doc.page_content.encode('utf-8').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass  # If decoding fails, leave the content as is

        return docs

    def query_multiple(self, prompt, course_codes=None, program_codes=None, num_codes=5):
        """
        Retrieve documents for multiple course or program codes.
        """
        docs = []
        seen = set()

        if course_codes:
            for code in set(course_codes):
                results = self.db.similarity_search(prompt, k=num_codes, filter={"course_code": code})
                for doc in results:
                    if doc.page_content not in seen:
                        # Ensure proper Unicode handling
                        try:
                            doc.page_content = doc.page_content.encode('utf-8').decode('utf-8')
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            pass
                        docs.append(doc)
                        seen.add(doc.page_content)

        if program_codes:
            for code in set(program_codes):
                results = self.db.similarity_search(prompt, k=num_codes, filter={"program_code": code})
                for doc in results:
                    if doc.page_content not in seen:
                        # Ensure proper Unicode handling
                        try:
                            doc.page_content = doc.page_content.encode('utf-8').decode('utf-8')
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            pass
                        docs.append(doc)
                        seen.add(doc.page_content)

        return docs

