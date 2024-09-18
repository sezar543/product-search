from typing import Annotated
from QA_WCD_chatbot_Streamlit_2 import setup_database, load_llm, qa_ENSEM_Run
from app_chatbot_feedback7 import qa_ENSEM_Run_SelfCheck
from fastapi import FastAPI, Body

db  = setup_database()
llm = load_llm()
app = FastAPI()

@app.get("/chatbot")
def response(query: Annotated[str, Body]):
  response = qa_ENSEM_Run_SelfCheck(query, llm = llm, db = db)
  return {"response": response[0], "check": response[1]}