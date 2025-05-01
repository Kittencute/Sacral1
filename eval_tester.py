from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
import numpy as np
import os

class EvalTester:
    def __init__(self, embed_model, log_dir="./eval_logs"):
        self.embed_model = embed_model
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(log_dir, f"eval_{timestamp}.txt")

    def compute_cosine_similarity(self, text1, text2):
        emb1 = self.embed_model.embed_query(text1)
        emb2 = self.embed_model.embed_query(text2)
        score = cosine_similarity([emb1], [emb2])[0][0]
        return score

    def compute_bertscore(self, candidate, reference):
        P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
        return float(F1[0])

    def log_evaluation(self, prompt, response, cosine, bert, reference):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"User Prompt: {prompt}\n\n")
            f.write(f"Reference Answer: {reference}\n\n")
            f.write(f"MDUBot Response: {response}\n\n")
            f.write(f"Cosine Similarity: {cosine:.4f}\n")
            if bert is not None:
                f.write(f"BERTScore (F1): {bert:.4f}\n")
            else:
                f.write("BERTScore (F1): N/A (no reference provided)\n")
            f.write("="*60 + "\n")
            
    def log_average_scores(self, avg_cosine, avg_bert):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n AVERAGE SCORES:\n")
            f.write(f"Cosine Similarity (avg): {avg_cosine:.4f}\n")
            f.write(f"BERTScore F1 (avg): {avg_bert:.4f}\n")
            f.write("="*60 + "\n")
