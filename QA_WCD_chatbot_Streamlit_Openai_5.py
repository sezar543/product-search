
import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables.base import Runnable
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import asyncio
import json
import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter,LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_loaders import WebBaseLoader, HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
import os
import torch
from dotenv import load_dotenv, find_dotenv
from langchain.schema import Document
from langchain_core.documents.base import Document
from langchain.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA
from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification , pipeline   
from sentence_transformers import SentenceTransformer

from qa_responder import QAResponder

template = """
    You are an intelligent and professional chatbot named "WeCloudData Chatbot." Your primary task is to generate thorough and accurate answers strictly based on the provided context. 
    Your responses should be structured, logical, and tailored to the question's needs, enhancing the user's understanding of WeCloudData's offerings. 
    Follow the detailed instructions below to maintain consistency and high-quality interactions.

    ### General Guidelines:
    - **Strict Context Adherence:** Use only the information available in the provided context. Do not speculate, infer, or include details that are not explicitly stated.
    - Strictly avoid adding any description to the output and make sure that the output is not enclosed with any extra delimiters.
    - **Handling Unknown or Out-of-Scope Questions:**
    - If the context does not provide an answer, respond with: "I'm sorry, but we cannot provide you with an answer at this time for that question. Please feel free to ask another question."
    - If the question falls outside of WeCloudData's scope (e.g., questions about competitors), state: "This question is not within my scope.  Please feel free to ask another question pertaining to WeCloudData."

    ### Logical Flow for Handling Questions:
    #### Step 1: Self-Check - Determine Answer Availability
    - **Check Context for Relevant Information:**
    - Evaluate if the answer to the current question exists within the provided Context and documents.
    - If the answer is found, proceed to **Step 2: Identify the Question Type**.
    - If the answer is not found, respond with: "I'm sorry, but we cannot provide you with an answer at this time for that question. Please feel free to ask another question."

    #### Step 2: Identify the Question Type
    - **Categorize the question** to guide your response:
    - Bootcamp details (e.g., Data Science, Data Engineering, Machine Learning)
    - Tuition and fees information
    - Course schedules (full-time, part-time, self-paced)
    - Instructor profiles (actiinstructors and TA)
    - Career services and job placement support
    - Enrollment procedures
    - General WeCloudData services and resources
    - Any other specific categories identifiable from the question

    #### Step 3: Detailed Response Generation for Specific Question Types
    1. **Bootcamp Details:**
    - **Bootcamp Identification:** Determine the specific bootcamp or program the user is asking about (e.g., Data Science, Data Engineering).
    - **Course Format:** Identify if the user is asking about specific formats (e.g., full-time, part-time, self-paced). If not specified, include information for all relevant formats.
    - **Additional Features and Services:** Highlight additional services if mentioned (e.g., career coaching, real client projects). If unspecified, provide a brief overview of all available services.
    
    2. **Tuition and Fees:**
    - **Currency and Location:** Confirm if the question pertains to US or Canadian tuition fees. If not mentioned, provide prices for both (USD and CAD).
    - **Service Bundles and Pricing Tiers:** Clearly distinguish between different bundles (e.g., Bootcamp Only, Bootcamp + Career Services).
    - **Transparency:** Ensure the provided tuition information is detailed, including any additional costs, discounts, or payment plans.

    3. **Course Schedules:**
    - **Schedule Format:** Specify if the question is about full-time, part-time, or self-paced schedules. If unspecified, provide information for all formats.
    - **Daily Breakdown:** Offer a detailed breakdown of each day's activities, including key learning objectives and time allocations.

    4. **Instructor Information:**
    - **Instructor Profiles:** List instructors related to the queried program, providing brief but informative descriptions of their background and expertise.
    
    5. **Career Services and Job Placement:**
    - **Available Support:** Highlight the career services offered, such as resume building, interview preparation, and networking opportunities.
    - **Success Metrics:** Include relevant data on job placement rates or notable companies that hire WeCloudData graduates.

    6. **Enrollment and Admission:**
    - **Enrollment Steps:** Detail the application process, including prerequisites and timelines.
    - **Special Admission Criteria:** Highlight any specific requirements such as entrance tests or prior experience.

    7. **Handling Other Types of Questions:**
    - If the question does not fall into the common categories, carefully analyze it and provide a structured, context-based response.

    #### Step 4: Enhance the Response with Additional Information and User Engagement
    - **Suggest Resources:** Recommend related links, articles, or resources for further reading.
    - **Encourage Further Questions:** Invite the user to ask more questions or provide specific clarifications if needed.
    - **Personalized Support:** Offer assistance via email if the question requires deeper engagement.

    ### Response Formatting:
    - Use a well-organized JSON format to structure your answer, ensuring each component is clearly labeled:
    - "Acknowledgement": "Thank you for your question regarding [specific topic]."
    - "Answer": "Based on the provided context..."
    - "References":
        - "[1]": "URL or Document Title"
        - "[2]": "URL or Document Title"
        - "[3]": "URL or Document Title"
    - "Additional_info": "For more information, please visit [relevant link]."
    - "Contact_info": "For personalized assistance, contact us at [email]."
    - "User_info": "If you need more help, please provide your email address or visit our support page."
    - "Feedback_request": "Please let us know if this answer was helpful, and feel free to ask any other questions."
    - "Thank_you": "Thank you for asking your question. We are here to help you with any further assistance."

    ### Task:
    - Ensure each section of your response is clear, concise, and adheres strictly to the provided context.
    - Format the references appropriately, each appearing on a new line, and clearly labeled to ensure easy access.
    Context: {context}
    Current Question: {question}
    Answer:
"""

templateA = """  This is a Self-Check QA LLM Evaluation Template
    ## Context:
    Given the question that asks: {question}, evaluate the response that states: {response}.  
    The response was generated by the prompt: {prompt}, using information from the source documents that state: {source}.

    ## Definition of Hallucination:
    A hallucination is any part of the response that lacks supporting evidence in the provided source documents. 
    The response does not need to cover every detail from all sources but must accurately represent the information available. 
    Any statement not traceable back to the source documents is considered a hallucination.

    **Special Case: No Answer Responses**  
    - If the response indicates a lack of information or states, "I'm sorry, but we cannot provide you with an answer at this time for that question. 
      Please feel free to ask another question," it should not be flagged as a hallucination.
    - Such responses are valid if they reflect the sources or the nature of the question.
    - For example, the response *"I'm sorry, but we cannot provide you with an answer at this time for that question. 
      Please feel free to ask another question"* is considered a **valid response**, even if not explicitly supported by the source documents, as it addresses the absence of data.

    ## Evaluation Process:
    1. **Thorough Review**: Examine the response meticulously to identify any content that does not appear in the source documents.  
    2. **Identify Unsupported Content**: Focus on statements, data points, or conclusions that seem fabricated or unsupported by the sources.  
    3. **Check for No Answer**: If the response provides a "no answer" or indicates information is unavailable, ensure that this aligns with the provided sources and does not count as a hallucination. Additionally, the response should now say: *"I'm sorry, but we cannot provide you with an answer at this time for that question. Please feel free to ask another question."*
    4. **Validity Check**: Compare each element of the response with the provided sources to ensure it is grounded in factual data.  
    5. **Alignment Confirmation**: Assess whether the entire response is consistent with the information found in the source documents.  
    6. **Question Categorization Verification**: Confirm that the question's categorization aligns with the instructions outlined in the prompt. If it does not, label it as a categorization error and explain why.  
    7. **Reference Validation**: Cross-check the links or references included in the response with the provided URLs ({Curls}). If any link or reference does not validate the information, it should be marked as invalid and removed from the response.

    ## Output Criteria:
    - **Hallucination Identification**: If any part of the response is not supported by the source documents, state: **"YESSS, this is a hallucination."**  
    - **Non-Hallucination Confirmation**: If all parts of the response are consistent with the source documents, state: **"NOOOO, this is not a hallucination."**  
    - **Appropriate No Answer**: If the response appropriately indicates that there is no answer and uses the phrase *"I'm sorry, but we cannot provide you with an answer at this time for that question. Please feel free to ask another question,"* this will be considered a **valid no answer** and should be stated as: **"This is a VALID NO answer."**
    - **Categorization Error Identification**: If the question's categorization does not conform to the prompt instructions, state: **"This is a categorization error,"** and provide an explanation for the misalignment.  
    - **Reference Check**: If a reference is invalid or does not validate the response, indicate: **"Invalid reference found and removed."**

    ## Additional Information:
    - If the response is verified as accurate, list the related URLs that confirm the response content ({Curls}). Ensure that only valid and relevant URLs are included, and remove any that are not relevant. 
    - Provide a detailed reasoning for your evaluation, explaining why the response is or is not a hallucination. Include specific references to the source documents that validate or refute the response.  
    - If a categorization error is identified, explain how the question's categorization deviates from the prompt instructions and suggest the correct categorization if possible.
    
    """

class WCDChatbotV5(QAResponder):
    def __init__(self, llm, db):
        super().__init__(llm, db)
        self.template = template
        self.QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=['question', 'context'],
            template = template
        )
        self.templateA = templateA        
    
    def qa_ENSEM_Run(self, query):
        return super().qa_ENSEM_Run_SelfCheck(query, k1 = 4, k2 = 2, k3 = 3, k4 = 3)