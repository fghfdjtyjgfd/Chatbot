# pip install tf_keras chromadb

from bs4 import BeautifulSoup
import os
import re
import requests
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def clean_text(txt):
    words_to_remove = ["สวัสดีครับ", "ครับ", "ค่ะ", "สวัสดีค่ะ"]
    pattern = "|".join(words_to_remove)
    return re.sub(pattern, "", txt.text) 

url = "https://www.agnoshealth.com/"
dataset = "dataset"
file = "data.txt"
file_path = os.path.join(dataset, file)
chroma = "Chroma"

if not os.path.exists(dataset):
    os.mkdir(dataset)

i = 1
while True:
    print(f"Page : {i}")
    page_content = requests.get(f"{url}forums/search?page={i}")
    soup = BeautifulSoup(page_content.content, "html.parser")
    post_links = soup.find_all("a", class_="undefined")
    if len(post_links) == 0:
        print("break")
        break

    for post_link in post_links:
        link = post_link.get('href')
        in_post_content = requests.get(url+link)
        in_post_soup = BeautifulSoup(in_post_content.content, "html.parser")
        doctor_ans = in_post_soup.find("p", class_="mt-4")
        if doctor_ans:
            doctor_ans = clean_text(doctor_ans)
            with open(file_path, "a") as f:
                f.write(doctor_ans+"\n\n")
        else:
            print("No doctor answer...")
    i += 1
        
loader = TextLoader(file_path)
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=400,
                                          chunk_overlap=80,
                                          separators=["\n\n", "\n", " "])
chunks = splitter.split_documents(documents)

if not os.path.exists(chroma):
    os.mkdir(chroma)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma.from_documents(chunks,
                           embedding=embedding,
                           persist_directory=chroma)