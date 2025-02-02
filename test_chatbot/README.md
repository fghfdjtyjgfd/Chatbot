## About The Project

# 🤖 AI Doctor Chatbot (RAG-Based)

A chatbot powered by **LangChain**, **OpenAI GPT-3.5**, **Hugging Face Embeddings**, **ChromaDB**, and **Streamlit**.  
This chatbot can **retrieve relevant medical knowledge** and **answer user queries based on past chat history**.

---

## 🚀 Features
✅ **Retrieval-Augmented Generation (RAG)** - Uses **ChromaDB** for relevant document retrieval.  
✅ **Memory-Powered Conversations** - Maintains **chat history** to improve response relevance.  
✅ **Thai Language Support** 🇹🇭 - Handles Thai medical queries naturally.  
✅ **Fast & Scalable** - Uses **Hugging Face embeddings** (`paraphrase-multilingual-MiniLM-L12-v2`).  
✅ **Interactive UI** - Built with **Streamlit** for an easy-to-use chat experience.  

---

![image](https://github.com/user-attachments/assets/942e0b48-a123-4a81-9128-a9d61a04b476)


## 📂 Project Structure
📄 app.py # Streamlit interface

📄 rag.py # Main RAG logic (Retriever, LLM, Memory) 

📄 requirements.txt # Python dependencies 

📄 README.md # Project documentation 

📂 dataset/ # Medical knowledge files storage

📄 data.txt # Medical knowledge corpus for retrieval 

📂 chroma/ # Vector database storage

📄 example.env # Add Openai api key here


## ⚙️ Installation

### **1️⃣ Install Dependencies**
install the required Python libraries:
```bash
pip install -r requirements.txt
```

### **2️⃣ Set Up .env File**
Rename an `example.env` file to `.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY="your openai api key"
```

### **3️⃣ Run ChromaDB Indexing**
Ensure the medical dataset is indexed into ChromaDB:
```bash
python3 rag.py
```

### **4️⃣ Start the Chatbot**
Using Streamlit:
```bash
streamlit run app.py
```
Then, open http://localhost:8501 in your browser.

---

📬 Contact

📧 Email: chaicharnsw@hotmail.com

🌐 GitHub: https://github.com/fghfdjtyjgfd
