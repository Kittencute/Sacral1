from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader
from uuid import uuid4


def load_courses():
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["course_code"] = record.get("kurskod").lower()
        metadata["course_name"] = record.get("name").lower()
        return metadata
    
    path = "mdu_data_url/course/courses.jsonl"
    loader = JSONLoader(file_path=path, json_lines=True, text_content=False, jq_schema=".", metadata_func=metadata_func)
    docs = loader.load()

    embed_model = OllamaEmbeddings(model="mxbai-embed-large")
    
    vector_store = Chroma(embedding_function=embed_model, persist_directory="./chroma") # Change persist_directory if you want another path for ChromaDB
    vector_store.add_documents(docs)
    print("Courses sucessfully added to DB")

def load_programs():
    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["program_code"] = record.get("programkod").lower()
        metadata["program_name"] = record.get("name").lower()
        return metadata
    
    path = "mdu_data_url/program/programs.jsonl"
    loader = JSONLoader(file_path=path, json_lines=True, text_content=False, jq_schema=".", metadata_func=metadata_func)
    docs = loader.load()

    embed_model = OllamaEmbeddings(model="mxbai-embed-large")
    
    vector_store = Chroma(embedding_function=embed_model, persist_directory="./chroma") # Change persist_directory if you want another path for ChromaDB
    vector_store.add_documents(docs)
    print("Programs sucessfully added to DB")


if __name__ == "__main__":
    load_courses()
    load_programs()
