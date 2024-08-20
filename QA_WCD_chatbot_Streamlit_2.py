
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.llms import HuggingFaceHub
import asyncio
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
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

import os
import torch
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import HuggingFaceEndpoint, HuggingFaceHub
from langchain import HuggingFacePipeline
from langchain.schema import Document
from langchain_core.documents.base import Document
from langchain.vectorstores import Chroma

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA

from transformers import AutoModel, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification , pipeline
from sentence_transformers import SentenceTransformer


def Extract_urls(sitemap_url): 
   response = requests.get(sitemap_url)
   sitemap_xml = response.content

   # Parse the sitemap
   soup = BeautifulSoup(sitemap_xml, "xml")

   # Extract all links
   links = []
   for link in soup.findAll("loc"):
       links.append(link.text)

   # Store the links in a variable
   return links

###-----------------------------------------------------------
def load_all_urls():
    sitemap = "https://weclouddata.com/sitemap_index.xml"
    # Extract all xmls 
    xmls=Extract_urls(sitemap)
    #print(len(xmls),"\n")
    #print(len(xmls),'xmls = ',xmls,"\n") #24 xmls

    list_URLsforXMLs=[]
    for i,xml in enumerate(xmls):
        URLsforXML=Extract_urls(xmls[i])
        list_URLsforXMLs.append(URLsforXML)
    #print(len(list_URLsforXMLs),"\n")
    #print('list_URLsforXMLs=',list_URLsforXMLs)  #24 list of url related to 24 xmls

    # courses and course-package
    all_urls=list_URLsforXMLs[4]+ list_URLsforXMLs[8] 

    #all_urls=[]
    #for URLsforXML in list_URLsforXMLs:
    #    all_urls += URLsforXML

    #print('all_urls = ',all_urls)
    #print(len(all_urls),"\n")

    def filter_image_urls(urls):
        filtered_urls = [url for url in urls if not (url.endswith('.jpeg') or url.endswith('.png') or url.endswith('.webp') or url.endswith('.gif') or url.endswith('.svg') or  url.endswith('.jpg'))]
        return filtered_urls

    urls = filter_image_urls(all_urls)
    print(len(urls),'urls = ',urls)
    return urls
###-----------------------------------------------------------
def load_clean_text_from_url(url):
  # Send a GET request to the webpage
  response = requests.get(url)

  # Check if the request was successful
  if response.status_code == 200:
      html_content = response.content

      # Parse the HTML content
      soup = BeautifulSoup(html_content, 'html.parser')
      # Find all body elements
      containers = soup.find_all('body')
      # Extract raw text for all body of Html
      for container in containers:
         raw_text = container.get_text(separator=' ', strip=True)
      # Clean the extracted text
      if raw_text:
         clean_text = ' '.join(raw_text.split())
         #print(clean_text)

  else:
     print("Failed to retrieve the webpage")

  return clean_text
###-----------------------------------------------------------

def create_hierarchy_header(headers):
    from collections import defaultdict

    def add_to_tree(tree, header, text):
        if header not in tree:
            tree[header] = {}
        tree = tree[header]
        return tree

    tree = {}
    levels = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4, 'H5': 5 }
    current_path = []

    for header in headers:
        level, text = header.split(': ', 1)
        level_num = levels[level]

        while current_path and levels[current_path[-1][0]] >= level_num:
            current_path.pop()

        if current_path:
            subtree = current_path[-1][1]
        else:
            subtree = tree

        subtree = add_to_tree(subtree, header, text)
        current_path.append((level, subtree))

    return tree
###-----------------------------------------------------------

def print_tree(tree, indent=0):
    # print tree of headers
    for key, subtree in tree.items():
        print(' ' * indent + key)
        print_tree(subtree, indent + 4)
###-----------------------------------------------------------

def create_hierarchy_target_header(headers):
    def add_to_tree(tree, level, text):
        if (level, text) not in tree:
            tree[(level, text)] = {}
        return tree[(level, text)]

    tree = {}
    levels = {'H1': 1, 'H2': 2, 'H3': 3, 'H4': 4, 'H5': 5 }
    current_path = []

    for header in headers:
        level, text = header.split(': ', 1)
        level_num = levels[level]

        while current_path and levels[current_path[-1][0]] >= level_num:
            current_path.pop()

        if current_path:
            subtree = current_path[-1][1]
        else:
            subtree = tree

        subtree = add_to_tree(subtree, level, text)
        current_path.append((level, subtree))

    return tree
###-----------------------------------------------------------

def find_path(tree, target, path=None):
    if path is None:
        path = []

    for (level, text), subtree in tree.items():
        new_path = path + [(level, text)]
        if text == target:
            return new_path
        found_path = find_path(subtree, target, new_path)
        if found_path:
            return found_path
    return None

###-----------------------------------------------------------

# Create Hierarchical_Paths for all headers
def find_hierarchical_header_Paths(hierarchy,headers_text):
    Hierarchical_header_Paths=[]
    for target_header in headers_text:
        path_to_target = find_path(hierarchy, target_header)

        if path_to_target:
           path_as_string = " -> ".join(["{}: {}".format(level, text) for level, text in path_to_target])
           Hierarchical_header_Path = f"This content belongs to the following hierarchical headers path; {path_as_string} "
           Hierarchical_header_Paths.append(Hierarchical_header_Path)  
        else:
           print("Header '{}' not found".format(target_header))
    
    #print(len(Hierarchical_header_Paths),Hierarchical_header_Paths[20])
    return Hierarchical_header_Paths
###-----------------------------------------------------------

def find_amount_of_dollars(headers, path):
    # Extract Tution fee for every bootcamp
    # Determine currency based on title
    if 'us' in path.lower():
        currency = ' USD '
    elif 'ca' in path.lower():
        currency = ' CAD '
    else:
        currency = ' '

    # Extract and format dollar amounts
    #Tuition = [f'${int(header)} {currency}' for header in headers if header.isdigit()]
    Tuition = ' , '.join([f' {int(header)} {currency} ' for header in headers if header.isdigit()])

    return Tuition
###-----------------------------------------------------------
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content={self.page_content!r}, metadata={self.metadata!r})"

###-----------------------------------------------------------
def is_word_at_start(text, word):
    # Strip any leading whitespace from the text and check if it starts with the word
    return text.strip().startswith(word)


def split_text_by_headers(text, headers):
    sections = []
    last_pos = 0
    for i, keyword in enumerate(headers):
        pos = text.find(keyword, last_pos)
        if pos != -1:
            if i > 0:
                sections.append(f" {text[last_pos:pos].strip()}")
            last_pos = pos
    sections.append(f" {text[last_pos:].strip()}")
    
    return sections
###-----------------------------------------------------------

def load_headers_creat_metadata(url):

    data = []
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    page_title = soup.title.string if soup.title else 'No title found'
    path_segment = url.split(".com")[1]

    # Extracting all headers and content
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4','h5'])
    
    for header in headers:
        header_tag = header.name.upper()
        header_text = header.get_text(strip=True)
        #header_text2 = header_text
        if header_text == "Drive Success with Interactive Learning Experience":
              header_text2 = header_text + " - Weekly Schedule "
        else:
              header_text2 = header_text

        header_final = f"{header_tag}: {header.get_text().strip()}"
        header_final2 = f"{header_tag}: {header_text2}"
        next_sibling = header.find_next_sibling()
        content = ''
        while next_sibling and next_sibling.name not in ['h1', 'h2', 'h3', 'h4','h5']:
            if next_sibling.name:
                content += next_sibling.get_text(strip=True) + " "
            next_sibling = next_sibling.find_next_sibling()
        
        data.append({
            'url': url,
            'path_segment': path_segment,
            'page_title': page_title,
            'header_tag' : header_tag,
            'header_text': header_text,
            'header_text2': header_text2,
            'header_final': header_final,
            'header_final2': header_final2,
            'content': content.strip()
        })

    #Convert to DataFrame and display
    df = pd.DataFrame(data)
    #df.head(20)
    headers_tag = df['header_tag'].tolist()
    headers_text = df['header_text'].tolist()
    headers_text2 = df['header_text2'].tolist()
    headers_final= df['header_final'].tolist()
    headers_final2= df['header_final2'].tolist()

    print(df.shape)

    text= load_clean_text_from_url(url)
    
    #print('URL_link:', url)
    #print("  ")
    #print(len(headers_final),'h=',headers_final)
    #print("<<<<---------->>>>")
    
    # Create hierarchy headers tree
    hierarchy_headers_tree = create_hierarchy_header(headers_final2)
    #print_tree(hierarchy_headers_tree)
    #print("<<<<--------------------------------------------------------------------->>>>")
    #print()
    # Get the dollar amounts
    Tuition = find_amount_of_dollars(headers_text, path_segment)

    # Creating the hierarchical tree
    hierarchy = create_hierarchy_target_header(headers_final2)

    Hierarchical_header_Paths = find_hierarchical_header_Paths(hierarchy,headers_text2)

    result = split_text_by_headers(text, headers_text)
    #print('result_sec_num=',type(result),len(result))

    #header_fee_paths = []
    metadata_documents = []
    for i, section in enumerate(result):
        if is_word_at_start(section, headers_text[i]):
                       
           #section_extention = f" URL :: {url} \n Title :: {page_title} \n Link :: {path_segment}\n Section {i} ::{Hierarchical_header_Paths[i]} >> {headers_text[i]} >>>\n {section} "
            section_extention = f"URL :: {url} >> Section {i} :: {Hierarchical_header_Paths[i]} >> {headers_text2[i]} >>> {section} "
            Hierarchical_header_Paths_S = Hierarchical_header_Paths[i]
            header_text_S = headers_text2[i]
        else:
           #section_extention=f" URL :: {url} \n Title :: {page_title} \n Link :: {path_segment}\n Section {i} ::{Hierarchical_header_Paths[i+1]} >> {headers_text[i+1]} >>>\n {section} "
            section_extention = f"URL :: {url} >> Section {i} :: {Hierarchical_header_Paths[i+1]} >> {headers_text2[i+1]} >>> {section} "
            Hierarchical_header_Paths_S = Hierarchical_header_Paths[i+1]
            header_text_S = headers_text2[i+1]
    #print(section_extention) 
    #print(">>>>>>>>>\n")
    
        metadata_documents.append(Document(
                    #page_content = section,
                    page_content = section_extention,
                    #page_content=f"Title :: {page_title}, Link :: {path_segment}, Section {i} :: {headers[i]} >>> {section} ",
                    metadata={"URL" : url, "Title" : page_title, "hierarchical headers path" : Hierarchical_header_Paths_S , "Header" : header_text_S , "Section number" :i }
                    ))


    header_fee_path = f"This content belongs to the following hierarchical headers of the tuition path; H1: {page_title} -> H5: Tuition Fee"
    #header_fee_paths.append(header_fee_path)
    metadata_documents.append(Document(
                    #page_content = section,
                    page_content = f"URL :: {url} >> Section {i+1} :: {header_fee_path} >> Tuition Fee  >>> {Tuition} ",
                    #page_content=f"Title :: {page_title}, Link :: {path_segment}, Section {i} :: {headers[i]} >>> {section} ",
                    metadata={"URL" : url , "Title" : page_title , "hierarchical headers path" : header_fee_path , "Header" : " Tuition Fee " , "Section number" :i+1 }
                    ))
    
    #print("i",i)
    #print("meta",len(metadata_documents))
    #print ('header=',result[0])
    return metadata_documents

###-----------------------------------------------------------
def Create_documents():
# Load headers tree for all URLs and creat metadata documents
    documents=[]
    urls= load_all_urls()
    for url in urls:
        all_headers =[]
        all_headers=load_headers_creat_metadata(url)
        for header in all_headers:
            documents.append(header)
            #print(doc)
            #print("--------")
    return documents

@st.cache_resource
def load_documents():
    output = Create_documents()  # Replace with your actual function
    return output

###-----------------------------------------------------------

def Find_all_hierarchical_headers_path():
    hierarchical_headers = []
    documents= load_documents()
    for doc in documents:
       hierarchical_headers.append(doc.metadata['hierarchical headers path'])
    return hierarchical_headers

@st.cache_resource
def load_all_hierarchical_headers_path():
    output = Find_all_hierarchical_headers_path()  # Replace with your actual function
    return output
###-----------------------------------------------------------

def Create_chunks():
    # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=300)

    # 'data' holds the text you want to split, split the text into documents using the text splitter.
    documents= load_documents()
    chunks = text_splitter.split_documents(documents)
    #print(len(chunks))
    return chunks
###-----------------------------------------------------------

def embed_chunks():
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
###-----------------------------------------------------------

def compute_db_vector_store():
    chunks =  Create_chunks()
    embeddings = embed_chunks()
    
    # Your actual computation or loading code
    persist_directory="./courses.db"    # MiniLM
    #persist_directory="./courses_mpn.db" # mpnet
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    #print(db._collection.count())
    #print(db._collection.get().keys())
    #print(db.get('embedding')["embeddings"])
    #print((len(db.get(include=['embeddings'])['embeddings'][10])))
    return db

@st.cache_resource
def load_db_vector_store():
    # Your code to determine the db vector store and get the output
    output = compute_db_vector_store()  # Replace with your actual function
    return output
###-----------------------------------------------------------

def Setup_llm():
    # Load environment variables
    load_dotenv('TokHap.env')
    HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

    model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

    # Set up the LLM from Hugging Face
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.2'
    mistral_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500})
    llm = mistral_llm
    return llm

@st.cache_resource
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

def fined_closest_header(query):

    if contains_word(query, "tuition"):

        hierarchical_headers=[
            'This content belongs to the following hierarchical headers of the tuition path; H1: Business Intelligence Bootcamp - WeCloudData -> H5: Tuition Fee',
            'This content belongs to the following hierarchical headers of the tuition path; H1: Data Science Bootcamp - WeCloudData -> H5: Tuition Fee',
            'This content belongs to the following hierarchical headers of the tuition path; H1: Data Engineering Bootcamp - WeCloudData -> H5: Tuition Fee',
            'This content belongs to the following hierarchical headers of the tuition path; H1: Machine Learning Engineering Bootcamp - WeCloudData -> H5: Tuition Fee']

    # List of headers
    headers = load_all_hierarchical_headers_path()

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

###-----------------------------------------------------------
def qa_ENSEM_Run(query):
    #Strictly avoid adding any description to the output and make sure that the output is not enclosed with any extra delimiters. 
    llm=load_llm()

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

    documents= load_documents()
    db = load_db_vector_store() # use only outputs
    closest_header = fined_closest_header(query)
    print(closest_header)

    k=6
    mmr1_retriever=db.as_retriever(search_kwargs={"k": k}, search_type = 'mmr')
    sim1_retriever=db.as_retriever(search_kwargs={"k": 2}, search_type = 'similarity')
    mmr3_retriever=db.as_retriever(search_kwargs={'k': k,'filter': {'hierarchical headers path': f'{closest_header}'}}, search_type = 'mmr')
    
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
    print('retriver_context= ',retriver_context )

    qa_ENSEM = RetrievalQA.from_chain_type(llm,
                                    chain_type='stuff',
                                    retriever= ensemble_retriever,
                                    return_source_documents=True,
                                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                                    )


    #result1 = qa_Rm.run(question=query, context=adjusted_context)
    result_Ensemble = qa_ENSEM({"query": query, "context":retriver_context })
    response_Ensemble = result_Ensemble['result']  # The chatbot's response
    #source_documents_Ensemble = result_Ensemble['source_documents']  
    #print("\nQuestion---->>>",response_Ensemble.split("Question:")[1])
    return response_Ensemble

#st.cache_data.clear()
#question=questions[0]
###-----------------------------------------------------------

# Function to update the history
def update_history(question, answer):
    st.session_state.history.append((question, answer))
    if len(st.session_state.history) > 0:
        st.session_state.history.pop(0)
###-----------------------------------------------------------
# Function to store feedback
def store_feedback(feedback):
    # Check if 'feedback' already exists in session_state
    # If not, then initialize it as an empty list
    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = []
    # Append the feedback to the session state
    st.session_state.feedback.append(feedback)

    # Here you would implement code to store feedback in S3
    # For demonstration, we're just printing the session state
    print(st.session_state['feedback'])
###-----------------------------------------------------------
def store_userinfo(user_contact):
    # Check if 'feedback' already exists in session_state
    # If not, then initialize it as an empty list
    if 'user_contact' not in st.session_state:
        st.session_state['user_contact'] = []
    # Append the feedback to the session state
    st.session_state.feedback.append(user_contact)

    # Here you would implement code to store feedback in S3
    # For demonstration, we're just printing the session state
    print(st.session_state['user_contact'])

###----------------------------------------------------------
def main():

    if "runnable" not in st.session_state:
        model = Setup_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
              (  "system",
                 """You are an intelligent and very knowledgeable chatbot operating for the site https://weclouddata.com/.
                    you are called "Weclouddata Chatbot".You answers in the conversation to maintain context and adjust your responses accordingly.
                    Provide detailed and accurate responses based on the documentation scope. 
                    Prevent hallucinations or providing unverified information. 
                    For out-of-scope questions, do not attempt to answer but inform the user that the question is beyond the current scope.

                    When you respond:
                    - Acknowledge the user's current question and any relevant context from previous interactions.
                    - Provide a comprehensive answer that covers all aspects of the question based on available documentation.
                    - Suggest additional information or resources on the topic when relevant.
                    - Direct the user to relevant links for more information.
                    - Ask for the user's email if more personalized help or follow-up is needed.
                    - Encourage further questions or clarifications to keep the conversation interactive.
                    - Format your response in JSON.

                    If a question is out of scope:
                    - Inform the user that the question is beyond the current scope of the chatbot,
                    - Suggest contacting support or visiting the documentation for more information.

                    
                    Current question: {{question}}
                    """
            ),
            ("human", "{question}"),
            ]
        )
        st.session_state.runnable = prompt | model | StrOutputParser()

        st.session_state.history = []

    # Streamlit app layout
    st.title("WeCloudData Chatbot System")
    st.write("Ask any question about WeCloudData and get an accurate and eloquent answer.")
   
    # Display the conversation history
    #for i, (q, a) in enumerate(st.session_state.history):
        #st.write(f"Q{i+1}: {q}")
        #st.write(f"A{i+1}: {a}")
    
    # Input from the user
    question = st.text_input("Enter your question:")

    if st.button("Get Answer... "):
        
        with st.spinner("Generating response..."):
            response_placeholder = st.empty()

            # Generate the response
            async def generate_response():
                msg = ""
                async for response in st.session_state.runnable.astream(
                    {"question": question},
                    config=RunnableConfig(callbacks=[]),
                ):
                        
                    response_Ens = qa_ENSEM_Run(question)
                    response=response_Ens.split("Answer:")[1]
                    msg += response
                    response_placeholder.write(msg)

                # Update the history after the response is generated
                # update_history(question, msg)

            #Use asyncio to run the async function
            asyncio.run(generate_response())


    # Feedback section
    st.write("### Provide Feedback Rate")
    #feedback = st.selectbox("Please provide your feedback rate (from 1-very good to 5-very bad):", [1, 2, 3, 4, 5])

    feedback_options = {
    "1. Excellent": 1,
    "2. Good": 2,
    "3. Average": 3,
    "4. Below Average": 4,
    "5. Poor": 5
}

    feedback_text = st.selectbox(
        "Please provide your feedback rate:", 
        list(feedback_options.keys())
    )

    feedback_value = feedback_options[feedback_text]

    if st.button("Submit Feedback"):
        #store_feedback("This is a sample feedback.")
        store_feedback(feedback_value)
        st.success("Thank you for your feedback!")
    
    
    # Functionality to review feedback (for demonstration purposes)
    #if st.button("Review Feedback"):
    #    st.write("Collected Feedback:")
    #    for fb in st.session_state.feedback:
    #        st.write(fb)

    st.write("### For more personalized help or follow-up")
    user_contact = st.text_input("Please provide your email address:")

    if st.button("Submit your Info"):
        store_userinfo(user_contact)
        st.success("Thank you for your info!")

             

if __name__ == "__main__":
    main()

