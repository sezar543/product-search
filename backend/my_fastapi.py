from typing import Annotated
from fastapi import FastAPI, Body
from pydantic import BaseModel
import sys
import os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from QA_WCD_chatbot_Streamlit_Openai_5 import WCDChatbotV5
from qa_responder import load_llm, QAResponder
# from QA_WCD_chatbot_Streamlit_Openai_4 import WCDChatbotV4
from data_pipeline import setup_database

COLLECTION_NAME = os.getenv("COLLECTION_NAME")

db  = setup_database()
llm = load_llm()
responder = WCDChatbotV5(llm = llm, db = db)
app = FastAPI()

@app.get("/chatbot")
def response(query: Annotated[str, Body]):
  print("query is ", query)
  response = responder.qa_ENSEM_Run(query)
  print("response is ", response)
  return {"response": response[0], "check": response[1]}

# Define input model
class TextData(BaseModel):
    input: str

@app.post("/test")
async def reverse_text(input: TextData):
    print("text received is = ", input.input)
    reversed_text = input.input[::-1]  # Reverse the input text
    return {"reversed_text": reversed_text}