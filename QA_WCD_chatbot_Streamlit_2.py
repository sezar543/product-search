
from dotenv import load_dotenv
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


def Find_all_hierarchical_headers_path(documents = None):
    #Maybe could use SQLAlchemy
    hierarchical_headers = []
    if not documents:
        documents = load_documents()
    for doc in documents:
       hierarchical_headers.append(doc.metadata['hierarchical headers path'])
    return hierarchical_headers


def load_all_hierarchical_headers_path(documents = None):
    output = Find_all_hierarchical_headers_path(documents)  # Replace with your actual function
    return output
###-----------------------------------------------------------

###-----------------------------------------------------------

def Setup_llm():
    # Load environment variables
    load_dotenv('TokHap.env')
    HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

    # Set up the LLM from Hugging Face
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.2'
    mistral_llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN, temperature = 0.1, max_new_tokens=500)
    llm = mistral_llm
    return llm

def load_llm():
    output = Setup_llm()  # Replace with your actual function
    return output

 #st.cache_data.clear()
###-----------------------------------------------------------

def all_questions ():
   questions = [ 
        "Who are the instructors for the Data Engineering Bootcamp?",   # 0
        "What is the weekly schedule for the Data Engineering Bootcamp?",# 1
        "What is the tuition for the Data Engineering Bootcamp",        # 2
        "Is there a self paced data engineering bootcamp?",             # 3
        "Are there any pre-requisites for enrolling in the Data Engineering Bootcamp?",# 4
        "Can students participate in live sessions during the self paced courses?",    # 5
        "What kind of mentorship is offered in the Data Science Bootcamp?",            # 6
        "What kind of projects are included in the AI Course: NLP/LLM?",               # 7
        "What are the differences between the Data Science with Python course and the Applied Machine Learning course?",# 8
        "What kind of real-world applications are included in the Data Engineering curriculum?", # 9
        "What kind of career support does WeCloudData offer to its students?",                   # 10
        "What are the main differences between self paced courses and bootcamp programs at WeCloudData?",# 11
        "What is the format of the TA support in self paced courses?",# 12
        "What is the duration of the Business Intelligence Bootcamp?",# 13
        "What is the focus of the DevOps Engineering Bootcamp?",      # 14
        "Who are the instructors for the Data Science Bootcamp?",     # 15
        "Who are the instructors for the Machine Learning Engineering Bootcamp?",   # 16
        "What is the weekly schedule for the Data Science Bootcamp",                # 17
        "What is the weekly schedule for the Machine Learning Engineering Bootcamp",# 18
        "What is the tuition for the Data Science Bootcamp",    # 19
        "What is the tuition for the Machine Learning Bootcamp",# 20
        "Are there any pre-requisites for enrolling in the Data Science Bootcamp?", #21
        "Are there any pre-requisites for the Machine Learning Bootcamp",           #22
        "What kind of mentorship is offered in the Data Engineering Bootcamp?",     #23
        "What kind of real-world applications are included in the Data Science curriculum?",     # 24
        "What kind of real-world applications are included in the Machine Learning curriculum?"]
         # 25
   query = questions[0]
   return query
###-----------------------------------------------------------

def contains_word(query, word):
    # Convert both the query and word to lowercase to make the check case-insensitive
    query_lower = query.lower()
    word_lower = word.lower()
    
    # Check if the word is in the query
    if word_lower in query_lower.split():
        return True
    else:
        return False
###-----------------------------------------------------------

def fined_closest_header(query, documents = None):
    #Maybe can use preprocessing
    if contains_word(query, "tuition"):
        #Should this be headers instead of hierarchical headers?
        headers=[
            'This content belongs to the following hierarchical headers of the tuition path; H1: Business Intelligence Bootcamp - WeCloudData -> H5: Tuition Fee',
            'This content belongs to the following hierarchical headers of the tuition path; H1: Data Science Bootcamp - WeCloudData -> H5: Tuition Fee',
            'This content belongs to the following hierarchical headers of the tuition path; H1: Data Engineering Bootcamp - WeCloudData -> H5: Tuition Fee',
            'This content belongs to the following hierarchical headers of the tuition path; H1: Machine Learning Engineering Bootcamp - WeCloudData -> H5: Tuition Fee']

    # List of headers
    headers = load_all_hierarchical_headers_path(documents)

    # Combine headers and query into a single list
    docs = headers + [query]

    # Vectorize the documents
    vectorizer = TfidfVectorizer().fit_transform(docs)
    vectors = vectorizer.toarray()

    # Compute cosine similarity between query and each header
    cosine_similarities = cosine_similarity(vectors[-1].reshape(1, -1), vectors[:-1]).flatten()

    # Find the index of the closest header
    closest_header_index = np.argmax(cosine_similarities)

    # Get the closest header
    closest_header = headers[closest_header_index]
    #print(type(closest_header),closest_header)
    return closest_header

def setup_database(test = True):
    embedding = embed_chunks()
    connection_string = get_test_connection_string() if test else get_connection_string()
    db = VectorDatabase(embedding = embedding, connection_string = connection_string)
    return db

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

#st.cache_data.clear()
#question=questions[0]
###-----------------------------------------------------------


