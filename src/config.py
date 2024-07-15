import os


class Config:
    def __init__(self):
        self.GEMINI_KEY = os.getenv("GEMINI_KEY")
        self.DB_URI = os.getenv("DB_URI")
        self.DB_NAME = os.getenv("DB_NAME")
        self.DB_COLLECTION = os.getenv("DB_COLLECTION")
        self.LLM_MODEL = os.getenv("LLM_MODEL")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

    def get_gemini_key(self):
        return self.GEMINI_KEY

    def get_db_uri(self):
        return self.DB_URI

    def get_db_name(self):
        return self.DB_NAME

    def get_db_collection(self):
        return self.DB_COLLECTION

    def get_llm_model(self):
        return self.LLM_MODEL

    def get_embedding_model(self):
        return self.EMBEDDING_MODEL
