from langchain_ollama import OllamaEmbeddings

class Retriever:
    def __init__(self, chroma_client, embed_model):
        self.embed_model = embed_model
        self.db = chroma_client

    # Query vector DB by course_code or program_code if provided, else fetch the 5 most relevant documents
    def query(self, prompt, course_code=None, program_code=None, num_codes=1):
        docs = []
        if course_code:
            docs = self.db.similarity_search(prompt, k=num_codes, filter={"course_code": course_code[0]})
        elif program_code:
            docs = self.db.similarity_search(prompt, k=num_codes, filter={"program_code": program_code[0]})
        elif not docs or not course_code or not program_code:
            docs = self.db.similarity_search(prompt, k=5)
            
        # Fix Unicode escape sequences in all retrieved documents 
        for doc in docs:
            doc.page_content = doc.page_content.encode().decode('unicode_escape')   

        return docs