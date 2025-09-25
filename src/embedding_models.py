import os
from sentence_transformers import SentenceTransformer

class Embedding_models:
    def __init__(self, model_name):
        self.model_name = model_name.split("/")[-1]  
        self.max_seq_length = None


    def get_embedding_model(self):
        if self.model_name in ["all-MiniLM-L6-v2", "bge-m3", "all-mpnet-base-v2", "embeddinggemma-300m"]:
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            embed_model = SentenceTransformer((f"./models/{self.model_name}" if os.path.isdir(f"./models/{self.model_name}") else (f"./{self.model_name}" if os.path.isdir(f"./{self.model_name}") else self.model_name)), device="cpu")
            self.max_seq_length = embed_model.get_max_seq_length()

            return embed_model
        
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")