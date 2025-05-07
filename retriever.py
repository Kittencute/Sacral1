from langchain_ollama import OllamaEmbeddings
import difflib
import re
import json

class Retriever:
    def __init__(self, chroma_client, embed_model):
        self.embed_model = embed_model
        self.db = chroma_client
        
    def parse_intent_response(self, intent_response: str) -> dict:
        """Extract and clean course names, program names, and keywords from LLM output."""
        fields = {
            "course_name": "",
            "program_name": "",
            "keywords": ""
        }

        # Extract fields from LLM response
        for line in intent_response.strip().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if key in fields:
                    fields[key] = value

        # Clean and return as lists
        return {
            "course_names": [s.strip() for s in fields["course_name"].split(",") if s.strip()],
            "program_names": [s.strip() for s in fields["program_name"].split(",") if s.strip()],
            "keywords": [s.strip() for s in fields["keywords"].split(",") if s.strip()],
        }
        
    # Query vector DB by course_code or program_code if provided, else fetch the 5 most relevant documents
    def query(self, user_prompt, intent_response):
        # Normalize input
        user_prompt = user_prompt.lower().strip()
        
        # Extract codes
        course_code = re.findall(r'\b[a-z]{2,3}\d{3}\b', user_prompt)
        program_code = re.findall(r'\b[a-z]{2,3}\d{2}\b', user_prompt)
        
        # Extract course and program names
        parsed = self.parse_intent_response(intent_response)
        
        # Regex pattern for course and program codes
        course_code_pattern = re.compile(r'^[a-zA-Z]{2,3}\d{3}$')
        program_code_pattern = re.compile(r'^[a-zA-Z]{2,3}\d{2}$')

        # Filter out any names that are just codes
        course_names = [
            name for name in parsed["course_names"]
            if not course_code_pattern.match(name.strip())
        ]

        program_names = [
            name for name in parsed["program_names"]
            if not program_code_pattern.match(name.strip())
        ]
        keywords = parsed["keywords"]
        
        print(course_code, program_code)
        print(course_names, program_names, keywords)
        
        # Semantic search
        code_c_docs = []
        code_p_docs = []
        sem_c_docs = []
        sem_p_docs = []
        
        # Search for each course code
        if course_code:
            for code in course_code:
                results = self.db.similarity_search(user_prompt, k=1, filter={"course_code": code})
                code_c_docs.extend(results)

        # Search for each program code
        if program_code:
            for code in program_code:
                results = self.db.similarity_search(user_prompt, k=1, filter={"program_code": code})
                code_p_docs.extend(results)
            
        # Search for each course name
        if course_names:
            for name in course_names:
                results = self.db.similarity_search(name, k=30)
                sem_c_docs.extend(results)

        # Search for each program name 
        if program_names:
            for name in program_names:
                results = self.db.similarity_search(name, k=30)
                sem_p_docs.extend(results)
        
        # Course name fuzzy match
        if course_names:
            target_course = course_names[0].lower()
            course_name_map = {
                doc.metadata.get("course_name", "").lower(): doc
                for doc in sem_c_docs if "course_name" in doc.metadata
            }
            closest_course = difflib.get_close_matches(target_course, list(course_name_map.keys()), n=1, cutoff=0.6)
            if closest_course:
                closest_name = closest_course[0]
                sem_c_docs = [doc for doc in sem_c_docs if doc.metadata.get("course_name", "").lower() == closest_name]

        # Program name fuzzy match
        if program_names:
            target_program = program_names[0].lower()
            program_name_map = {
                doc.metadata.get("program_name", "").lower(): doc
                for doc in sem_p_docs if "program_name" in doc.metadata
            }
            closest_program = difflib.get_close_matches(target_program, list(program_name_map.keys()), n=1, cutoff=0.6)
            if closest_program:
                closest_name = closest_program[0]
                sem_p_docs = [doc for doc in sem_p_docs if doc.metadata.get("program_name", "").lower() == closest_name]
                     
        # Combine all docs
        docs = code_c_docs + sem_c_docs + code_p_docs + sem_p_docs
        
        # Remove duplicates      
        seen = set()
        unique_docs = []
        for doc in docs:
            key = doc.metadata.get("course_code") or doc.metadata.get("program_code")
            if key and key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        docs = unique_docs    
                
        # Fix Unicode escape sequences in all retrieved documents 
        for doc in docs:
            doc.page_content = doc.page_content.encode().decode("unicode_escape") 
            
        blocks = []

        for doc in docs:
            meta = doc.metadata
            code = meta.get("course_code") or meta.get("program_code", "")
            name = meta.get("course_name") or meta.get("program_name", "")
            label = "COURSE" if "course_code" in meta else "PROGRAM"

            try:
                # Try to load and pretty-print JSON
                content_json = json.loads(doc.page_content)
                content_pretty = json.dumps(content_json, indent=2, ensure_ascii=False)
            except Exception:
                # If it's not valid JSON, just use it as-is
                content_pretty = doc.page_content.strip()

            blocks.append(
                f"--- {label}: {code or '[No Code]'} ---\n"
                f"Name: {name or '[No Name]'}\n\n"
                f"{content_pretty}"
            )

        return docs