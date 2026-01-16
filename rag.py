# A short guide to the EU documentations on RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(
    level=logging.WARNING,  # Only show WARNING and above for root logger
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set your own logger to INFO level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress noisy third-party loggers
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class RAGSystem(BaseModel):
    file_path: str | Path
    prompt_template: str
    db_name: str = "chroma_db"  # Default value
    chunk_size: int = Field(default=1000, ge=100, le=10000)  # With constraints
    chunk_overlap: int = Field(default=100, ge=0)
    model_embedding: str = "gemini-embedding-001"
    model_llm: str = "gemini-2.5-flash"
    search_type: str = "similarity"
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"k": 3})
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    def _load_data(self) -> list[Document]:
        """Load data from file path"""

        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            
            logger.info(f"Loading PDF file from {self.file_path}")
            loader = PyPDFLoader(self.file_path)
            data = loader.load()
            #logger.info(f"Loaded {len(data)} documents")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _split_data(self) -> list[Document]:
        """Split data into chunks"""
        try:
            logger.info(f"Splitting data into chunks")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            chunks = text_splitter.split_documents(self._load_data())
            #logger.info(f"Split {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Error splitting data: {e}")
            return None
    
    def _embed_data_google(self) -> GoogleGenerativeAIEmbeddings:
        """Embed data"""
        try:
            client = genai.Client()
            embeddings = GoogleGenerativeAIEmbeddings(model=self.model_embedding)
            #logger.info(f"Embedded {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            print(f"Error embedding data: {e}")
            return None
    
    def _store_data(self) -> Chroma:
        """Store data in Chroma"""
        try:
            if os.path.exists(self.db_name):
                logger.info(f"Loading existing Chroma database from {self.db_name}")
                vectorstore = Chroma(
                    persist_directory=self.db_name,
                    embedding_function=self._embed_data_google()
                )
                #logger.info(f"Loaded existing Chroma database from {self.db_name}")
            else:
                logger.info(f"Creating new Chroma database in {self.db_name}")
                vectorstore = Chroma.from_documents(
                documents=self._split_data(), 
                embedding=self._embed_data_google(),
                persist_directory=self.db_name)
                #logger.info(f"Created new Chroma database in {self.db_name}")
            
            #vectorstore.persist()
            return vectorstore
        except Exception as e:
            print(f"Error storing data: {e}")
            return None
        
    
    def _retriever(self) -> BaseRetriever:
        """Retrieve data from Chroma"""
        try:
            logger.info(f"Retrieving data from Chroma")
            retriever = self._store_data().as_retriever(
                search_type=self.search_type,
                search_kwargs=self.search_kwargs
            )
            return retriever
        except Exception as e:
            print(f"Error retrieving data: {e}")
            return None
    
    def _prompt_template(self) -> ChatPromptTemplate:
        """Prompt template"""
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        return prompt_template
    
    def _llm_google(self) -> ChatGoogleGenerativeAI:
        """LLM"""
        try:
            #logger.info(f"LLM")
            llm = ChatGoogleGenerativeAI(model=self.model_llm, temperature=self.temperature)
            return llm
        except Exception as e:
            print(f"Error LLM: {e}")
            return None
    
    @staticmethod
    def format_docs(docs : list[Document]) -> str:
        """Format documents"""
        return "\n\n".join(doc.page_content for doc in docs)

    def _chain(self) -> Runnable:
        """Chain"""
        try:
            logger.info(f"Chain")
            return (
            {"context": RunnableLambda(lambda x: x["question"]) | self._retriever() | self.format_docs
            , "question": RunnablePassthrough()} 
            | self._prompt_template() 
            | self._llm_google() 
            | StrOutputParser()
            )
        except Exception as e:
            print(f"Error chain: {e}")
            return None

    def ask(self, question : str) -> str:
        """Ask a question"""
        try:
            result = self._chain().invoke({"question": question})
            logger.info("-" * 50)
            logger.info(f"Question: {question}")
            logger.info(f"Answer: {result}")
        except Exception as e:
            print(f"Error: {e}")
            return None
        return result
