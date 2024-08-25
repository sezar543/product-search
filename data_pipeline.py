from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from langchain.schema import Document
from langchain_core.documents.base import Document
import requests

from vector_database import VectorDatabase


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


def load_documents():
    output = Create_documents()  # Replace with your actual function
    return output


def Create_chunks():
    # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=300)

    # 'data' holds the text you want to split, split the text into documents using the text splitter.
    documents= load_documents()
    chunks = text_splitter.split_documents(documents)
    return chunks


###-----------------------------------------------------------
# Above is all the methods required for setting up the database
# Can really customize the bottom - add in any embedding
#TODO: complete the below
def get_connection_string():
    pass

def get_test_connection_string():
    pass

###------------------------------------------------------------
# Below is the methods needed to load documents and add it to the vector databse
# TODO: complete the below method

def add_documents_to_database(db: VectorDatabase):
    documents = load_documents()
    return db.add_documents(documents)

if __name__ == '__main__':
    embeddings = embed_chunks()
    # connection_string = get_connection_string()
    connection_string = get_test_connection_string()
    #Assumes that a database has already been created 
    db = VectorDatabase(embedding=embeddings, connection_string=connection_string)

    chunks = Create_chunks()
    db.add_documents(chunks)
