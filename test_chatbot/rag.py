import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.prompts import MessagesPlaceholder

load_dotenv()

dataset = "dataset"
file = "data.txt"
file_path = os.path.join(dataset, file)
chroma = "Chroma"

contextualize_system_prompt = """
ให้ประวัติการสนทนาและคำถามล่าสุดของผู้ใช้  
ซึ่งอาจมีการอ้างอิงถึงบริบทจากประวัติการสนทนา  
ให้สร้างคำถามใหม่ที่สามารถเข้าใจได้โดยไม่ต้องมีประวัติการสนทนา  
ห้ามตอบคำถาม เพียงแค่ปรับเปลี่ยนรูปแบบของคำถามหากจำเป็น  
หากไม่จำเป็น ให้ส่งคำถามเดิมกลับไป  

{chat_history}
"""

system_prompt = """
คุณเป็นแพทย์ที่ให้คำแนะนำเกี่ยวกับโรคและอาการของผู้ป่วย  
ใช้ข้อมูลที่ได้รับมาเป็นพื้นฐานในการตอบคำถาม  
หากไม่มีข้อมูลเพียงพอ ให้บอกว่าไม่สามารถให้คำตอบได้  
โปรดตอบสั้นๆ ไม่เกินสามประโยค เป็นภาษาไทย

{context}
"""

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = Chroma(embedding_function=embedding,
            persist_directory=chroma)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
retriever = db.as_retriever(search_type="mmr",
                            search_kwargs={"k":7, "fetch_k": 15, "lambda_mult": 0.6})

retriever_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
histroy_aware_retriever = create_history_aware_retriever(llm=llm,
                                                         retriever=retriever,
                                                         prompt=retriever_prompt)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(histroy_aware_retriever, qa_chain)

