from langchain.embeddings import HuggingFaceEmbeddings

class VectorDatabase():
  def __init__(self, embedding = None) -> None:
    if embedding:
      self.embedding = embedding
    else:
      self.embedding = self.get_embeddings()
    

  def get_embeddings(self):

    embedding = HuggingFaceEmbeddings()
    return embedding
  def invoke(self, query):
    pass