from typing import Annotated
from QA_WCD_chatbot_Streamlit_2 import load_llm, qa_ENSEM_Run, setup_database
from fastapi import FastAPI, Body

db  = setup_database()
llm = load_llm()
app = FastAPI()

@app.get("/chatbot")
def response(query: Annotated[str, Body]):
  response = qa_ENSEM_Run(query, llm = llm, db = db)
  #response = "hello world, I am wcd chatbot, this is your question: " + query
  return {"response": response}