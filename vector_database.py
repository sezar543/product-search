import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
import psycopg2
from langchain_core.documents.base import Document

load_dotenv()
COLLECTION_NAME = os.environ["COLLECTION_NAME"]

class VectorDatabase():
  def __init__(self, embedding = None, connection_string = None, collection_name="") -> None:
    self.collection_name = collection_name
    self.connection_string = connection_string
    if embedding:
      self.embedding = embedding
    else:
      self.embedding = self.get_embeddings()
    
    self.db = PGVector(embeddings=self.embedding, connection=self.connection_string, collection_name=COLLECTION_NAME)
    self.documents = None
  
  def get_db(self):
    return self.db
    

  def get_embeddings(self):
    # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-MiniLM-L6-v2"   #384
    #modelPath = "sentence-transformers/all-mpnet-base-v2"   #768

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device': 'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to True
    encode_kwargs = {'normalize_embeddings': True}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
         model_name=modelPath,     # Provide the pre-trained model's path
         model_kwargs=model_kwargs, # Pass the model configuration options
         encode_kwargs=encode_kwargs # Pass the encoding options
            )
    return embeddings
  
  def add_documents(self, documents):
    return self.db.add_documents(documents)
  
  def as_retriever(self, **kwargs):
    return self.db.as_retriever(**kwargs)


  def invoke(self, query, search_type, **kwargs):
    return self.db.search(query=query, search_type=search_type, **kwargs)
  
  def get_all_documents(self):
    with psycopg2.connect(self.connection_string) as conn:
      curr = conn.cursor()
      table_name = f"{self.collection_name}_langchain_pg_embedding" if self.collection_name else "langchain_pg_embedding"
      curr.execute(f"SELECT document, cmetadata FROM {table_name};")
      return [Document(page_content=text, metadata = metadata) for text, metadata in curr.fetchall()]