from langchain_ollama import OllamaEmbeddings
import difflib
import re
import json

class Retriever:
    def __init__(self, chroma_client, embed_model):
        self.embed_model = embed_model
        self.db = chroma_client
        
    def fuzzy_word(self, docs, name, field_name, cutoff=0.6):
        name_map = {}
        for doc in docs:
            key = doc.metadata.get(field_name, "").lower()
            if key:
                name_map.setdefault(key, []).append(doc)

        target = name.lower()
        closest = difflib.get_close_matches(target, list(name_map.keys()), n=1, cutoff=0.7)

        if closest:
            return name_map[closest[0]]
        else:
            # fallback: return everything if nothing matched above cutoff
            all_docs = []
            for key in name_map:
                all_docs.extend(name_map[key])
            return all_docs
                
    def extract_name(self, user_prompt):
        # Normalize text
        cleaned = user_prompt.lower()

        # Define blacklist of filler words
        blacklist = {
            "what", "is", "can", "i", "in", "the", "and", "or", "of", "about", "to", "for",
            "you", "it", "on", "a", "an", "do", "does", "that", "with", "from", "be", "any"
        }

        # Tokenize first and remove filler words
        words = re.findall(r'\b[a-zåäö]{2,}\b', cleaned)
        filtered_words = [w for w in words if w not in blacklist]

        # Rejoin to a cleaned string for name pattern extraction
        filtered_text = " ".join(filtered_words)

        # Extract possible name-like sequences (up to 4 words)
        candidates = re.findall(r'\b[a-zåäö]{3,}(?:\s+[a-zåäö]{2,}){0,3}\b', filtered_text)

        return candidates  
    def search_semantic_name(self, db, name): 
        # Perform a semantic search for the name
        docs = db.similarity_search(name, k=30)
        return docs     

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

        print(user_prompt)
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
                results = self.search_semantic_name(self.db, name)
                results = self.fuzzy_word(results, name, "course_name")
                sem_c_docs.extend(results)

        # Search for each program name 
        if program_names:
            for name in program_names:
                results = self.search_semantic_name(self.db, name)
               # results = self.fuzzy_word(results, name, "program_name")
                sem_p_docs.extend(results)
                     
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
            
        for doc in docs:
            meta = doc.metadata
            course_code = meta.get("course_code")
            course_name = meta.get("course_name")
            program_code = meta.get("program_code")
            program_name = meta.get("program_name")

            if course_code:
                print(f"Course Code: {course_code}\tCourse Name: {course_name or 'N/A'}")
            elif program_code:
                print(f"Program Code: {program_code}\tProgram Name: {program_name or 'N/A'}")

        return docs