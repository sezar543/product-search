from langchain.embeddings import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

class VectorDatabase():
  def __init__(self, embedding = None, connection_string = None) -> None:
    self.connection_string = connection_string
    if embedding:
      self.embedding = embedding
    else:
      self.embedding = self.get_embeddings()
    
    self.db = PGVector(embeddings=self.embedding, connection=self.connection_string)
  
  def get_db(self):
    return self.db
    

  def get_embeddings(self):
    embedding = HuggingFaceEmbeddings()
    return embedding
  
  def add_documents(self, documents):
    return self.db.add_documents(documents)


  def invoke(self, query, search_type, **kwargs):
    return self.db.search(query=query, search_type=search_type, **kwargs)