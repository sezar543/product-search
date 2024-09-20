from typing import Annotated
from qa_responder import load_llm, QAResponder
from data_pipeline import setup_database
from fastapi import FastAPI, Body

db  = setup_database()
llm = load_llm()
responder = QAResponder(llm = llm, db = db)
app = FastAPI()

@app.get("/chatbot")
def response(query: Annotated[str, Body]):
  response = responder.qa_ENSEM_Run_SelfCheck(query)
  return {"response": response[0], "check": response[1]}