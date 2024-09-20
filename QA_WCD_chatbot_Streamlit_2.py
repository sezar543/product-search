
from dotenv import find_dotenv, load_dotenv
from data_pipeline import embed_chunks, get_connection_string, get_test_connection_string, load_documents
#Below is deprecated, need to use langchain_huggingface.llms.huggingface_endpoint.HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpoint
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

import os

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA

from vector_database import VectorDatabase
from qa_responder import QAResponder

class WCDChatbotV2(QAResponder):
    def qa_ENSEM_Run_SelfCheck(query):
        return qa_ENSEM_Run(query, llm = self.llm, db = self.db), ""

###-----------------------------------------------------------
def qa_ENSEM_Run(query, llm, db):
    #Strictly avoid adding any description to the output and make sure that the output is not enclosed with any extra delimiters. 
    # llm = load_llm()
    # Define the base template
    template = """You are an intelligent chatbot. you are called "Weclouddata Chatbot"
    Based on the provided context and source documents, please generate a detailed and thorough answer to the question.
    Refrain from inventing any information not explicitly stated in the Context.
    If uncertain, simply state that you do not know instead of speculating.
    If you do not find the answer in the result, please state 'There is No answer.'
    If there is 'pt' in path segment it means part-time and 'ft' in path segment it means full-time.
    If there is 'us' in path segment it means US and 'ca' in path segment it means CANADA. 
    Ensure that the response includes comprehensive explanations, relevant details, and is organized logically.

    For example:
    - If the question pertains to a bootcamp, ensure it directly addresses a specific type, such as Data Science Bootcamp, Data Engineering Bootcamp, Machine Learning (ML) Bootcamp, etc. 
    - Please provide an answer based on the correct type of bootcamp, making distinctions between similar options, such as Big Data Engineering Bootcamp and Data Engineering Bootcamp.
    - If the question pertains to a weekly schedule, provide a detailed breakdown of activities for each day, arranged by day and hour.
    - If the question pertains to instructors, provide the names of all instructors and an abstract of detailed information, arranged by name.
    - If the question pertains to the types of bootcamps, you should check the different existing kinds: full-time, part-time, and self-paced bootcamps.
    - If the question pertains to the tuition of a bootcamp, check the different types available—full-time, part-time, and self-paced—and provide the costs in both USD (US) and CAD (CA).
    - If the question involves a complex topic, break down the explanation into clear sections and subsections, covering all necessary aspects in summary.
    - Include examples, analogies, or case studies where applicable to enhance understanding.
    Please provide a well-structured, detailed, and insightful response.

    When you respond:
    - Acknowledge the user's current question and any relevant context.
    - Provide a comprehensive answer that covers all aspects of the question based on available documentation,
    - Suggest additional information or resources on the topic when relevant. Also, provide all URL sources existing in the Context that were used to respond.
    - Direct the user to relevant links for more information.
    - Ask for the user's email if more personalized help or follow-up is needed.
    - Encourage further questions or clarifications to keep the conversation interactive.
    - Format your response in JSON.

    If a question is out of scope:
    - Inform the user that the question is beyond the current scope of the chatbot,
    - For exaple if user askd about WeCloudData competitors, you have to answer "this question is not in my scope".
    - Suggest contacting support or visiting the documentation for more information.

    Task: Please format the references with each one on a separate line, preceded by a short title. The section number in the References structure is not necessary and should be removed.
    Output Format: The output should be structured as a JSON string, with each key presented on separated lines and each key-value pair prefixed by a bullet point. The required keys are as follows:
    - "Acknowledgement": "Your question about topic..."
    - "Answer": "Based on the documentation..."
    - "References": 
        - "[1]": "url1"  
        - "[2]": "url2"  
        - "[3]": "url3"  
    - "Additional_info": "For more details, visit..."
    - "Contact_info": "For personalized assistance, email us at..."
    - "User_info": "For more assistance, provide your email address or..."
    - "Feedback_request": "Please provide feedback on this response."
    
      
    Context: {context}
    current Question: {question}
    Answer: """

    # Define the QA chain prompt template
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=['question', 'context'],
        template = template
    )

    #We want to get the documents from the database not testing 
    #TODO: need to time below
    documents= db.get_all_documents()
    #replacing below with adding database in the arguments
    # db = setup_database() 
    closest_header = fined_closest_header(query, documents=documents)

    k=6
    mmr1_retriever=db.as_retriever(search_kwargs={"k": k}, search_type = 'mmr')
    sim1_retriever=db.as_retriever(search_kwargs={"k": 2}, search_type = 'similarity')
    #Somehting wrong with below
    mmr3_retriever=db.as_retriever(search_kwargs={'k': k,'filter': {'hierarchical_headers_path': f'{closest_header}'}}, search_type = 'mmr')
    #Below is a bit wierd, since we wanna get documents from the database
    #Why not get the below from as_retriever method
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

    qa_ENSEM = RetrievalQA.from_chain_type(llm,
                                    chain_type='stuff',
                                    retriever= ensemble_retriever,
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                                    )


    #result1 = qa_Rm.run(question=query, context=adjusted_context)
    result_Ensemble = qa_ENSEM({"query": query, "context":retriver_context })
    response_Ensemble = result_Ensemble['result']  # The chatbot's response
    return response_Ensemble


