# A short guide to the EU documentations on RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class RAGSystem:
    def __init__(self, file_path, prompt_template, db_name="chroma_db", chunk_size=1000, chunk_overlap=100, 
    model_embedding="gemini-embedding-001", model_llm="gemini-2.5-flash", search_type="similarity", search_kwargs={"k": 3},
    temperature=0.0):
    
        self.file_path = file_path
        self.prompt_template = prompt_template
        self.db_name = db_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_embedding = model_embedding
        self.model_llm = model_llm
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.temperature = temperature

    def _load_data(self):
        """Load data from file path"""
        loader = PyPDFLoader(self.file_path)
        data = loader.load()
        return data
    
    def _split_data(self):
        """Split data into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(self._load_data())
        return chunks
    
    def _embed_data_google(self):
        """Embed data"""
        client = genai.Client()
        embeddings = GoogleGenerativeAIEmbeddings(model=self.model_embedding)
        return embeddings
    
    def _store_data(self):
        """Store data in Chroma"""
        if os.path.exists(self.db_name):
            vectorstore = Chroma(
            persist_directory=self.db_name,
            embedding_function=self._embed_data_google()
        )
        else:
            vectorstore = Chroma.from_documents(
            documents=self._split_data(), 
            embedding=self._embed_data_google(),
            persist_directory=self.db_name
        )
        #vectorstore.persist()
    
        return vectorstore
    
    def _retriever(self):
        """Retrieve data from Chroma"""
        retriever = self._store_data().as_retriever(
            search_type=self.search_type,
            search_kwargs=self.search_kwargs
        )
        return retriever
    
    def _prompt_template(self):
        """Prompt template"""
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        return prompt_template
    
    def _llm_google(self):
        """LLM"""
        llm = ChatGoogleGenerativeAI(model=self.model_llm, temperature=self.temperature)
        return llm
    
    @staticmethod
    def format_docs(docs):
        """Format documents"""
        return "\n\n".join(doc.page_content for doc in docs)

    def _chain(self):
        """Chain"""
        return (
        {"context": RunnableLambda(lambda x: x["question"]) | self._retriever() | self.format_docs
        , "question": RunnablePassthrough()} 
        | self._prompt_template() 
        | self._llm_google() 
        | StrOutputParser()
    )

    def ask(self, question):
        """Ask a question"""
        result = self._chain().invoke({"question": question})
        print(result)
        return result
