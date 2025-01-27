from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.llms import HuggingFaceHub

import json

import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from huggingface_hub import HfApi, HfFolder
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter,LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_loaders import WebBaseLoader, HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

import os
import torch
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import HuggingFaceEndpoint, HuggingFaceHub
from langchain import HuggingFacePipeline
from langchain.schema import Document
from langchain_core.documents.base import Document

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification , pipeline   #AutoModel
from sentence_transformers import SentenceTransformer


###-----------------------------------------------------------
def qa_ENSEM_Run_SelfCheck(query, llm, db):
    #Strictly avoid adding any description to the output and make sure that the output is not enclosed with any extra delimiters. 

    # Define the base template
    template = """
    You are an intelligent and professional chatbot named "WeCloudData Chatbot." Your primary task is to generate thorough and accurate answers strictly based on the provided context. 
    Your responses should be structured, logical, and tailored to the question's needs, enhancing the user's understanding of WeCloudData's offerings. 
    Follow the detailed instructions below to maintain consistency and high-quality interactions.

    ### General Guidelines:
    - **Strict Context Adherence:** Use only the information available in the provided context. Do not speculate, infer, or include details that are not explicitly stated.
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
    - Instructor profiles
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

    # Define the QA chain prompt template
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=['question', 'context'],
        template = template
    )

    documents= db.get_all_documents()
    closest_header = fined_closest_header(query, documents = documents)

    k=6
    mmr1_retriever=db.as_retriever(search_kwargs={"k": k}, search_type = 'mmr')
    sim1_retriever=db.as_retriever(search_kwargs={"k": 2}, search_type = 'similarity')
    mmr3_retriever=db.as_retriever(search_kwargs={'k': k,'filter': {'hierarchical_headers_path': f'{closest_header}'}}, search_type = 'mmr')
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 4
    #retriver_context= mmr1_retriever+sim1_retriever+mmr3_retriever+bm25_retriever

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, mmr1_retriever, sim1_retriever,mmr3_retriever], weights=[0.5,0.5,0.5,0.5])
    
    #retriver_context1 = ensemble_retriever.get_relevant_documents(query=query)
    retriver_context1 = ensemble_retriever.invoke(query)    
    retriver_context2 = sim1_retriever.invoke(query)
    retriver_context3 = mmr1_retriever.invoke(query)
    retriver_context4 = mmr3_retriever.invoke(query)
    retriver_context  = retriver_context1 + retriver_context2 + retriver_context3 + retriver_context4
    #retriver_context = retriver_context1 
    #print('retriver_context= ',retriver_context )

    qa_ENSEM = RetrievalQA.from_chain_type(llm,
                                    chain_type='stuff',
                                    retriever= ensemble_retriever,
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                                    )


    #result1 = qa_Rm.run(question=query, context=adjusted_context)
    result_Ensemble = qa_ENSEM({"query": query, "context":retriver_context })
    response_Ensemble = result_Ensemble['result']  # The chatbot's response
    #print("\nQuestion---->>>",response_Ensemble.split("Question:")[1])
    source_documents_Ensemble = result_Ensemble['source_documents']
    # Extract URLs
    extracted_urls = extract_urls(source_documents_Ensemble)
     
    #st.cache_data.clear()
    #question=questions[0]

    templateA=""" Given the question that asks {question}, evaluate the response that states {response}.  
    The response was generated by the prompt that states {prompt}, 
    using the information from the source documents that state {source}.  

    Definition of Hallucination:  
    A hallucination is any part of the response that does not have supporting evidence in the provided source documents.  
    The response does not need to cover all information from every source, 
    but it must accurately represent what is found in the sources.

    Evaluation Process:  
    1. Conduct a thorough review of the response to identify any content not found in the source documents.  
    2. Pay close attention to any statements, data points, or conclusions that appear unsupported or fabricated.  
    3. Compare the response with the provided sources to confirm its validity.  
    4. Determine if the response is entirely aligned with the information found in the sources.

    Output Criteria:  
    - If any part of the response is not found in the source documents, 
    state: "Yesss, This is a hallucination" and output `1`.  
    - If all parts of the response are consistent with the source documents, 
    state: "NOOOO, this is not a hallucination" and output `0`.  

    Additional Information:  
    - If the response is verified as accurate (output `0`), 
    provide the related URLs that validate the response content ({Curls}).  
    - Include detailed reasoning to support the evaluation outcome, explaining why the response is or is not a hallucination.
    
    """
    
    template_string = templateA

    # Format the template string
    formatted_template = templateA.format(
        question = query,
        response = response_Ensemble,
        prompt = QA_CHAIN_PROMPT,
        source = source_documents_Ensemble,
        Curls = extracted_urls
    )

    # Assuming `llm` is a function that processes the formatted_template
    check = llm(formatted_template)
    return response_Ensemble, check