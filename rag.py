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
    """
    A Retrieval-Augmented Generation (RAG) system for querying PDF documents.
    
    This class implements a complete RAG pipeline that loads PDF documents, splits them
    into chunks, generates embeddings using Google's Gemini models, stores them in a
    ChromaDB vector database, and retrieves relevant context to answer user queries
    using a language model.
    
    The system uses Pydantic for automatic validation of configuration parameters,
    ensuring type safety and constraint validation at runtime.
    
    Attributes:
        file_path: Path to the source PDF document to process.
        prompt_template: Template string for LLM prompts. Must contain {context}
                         and {question} placeholders.
        db_name: Name of the ChromaDB database directory. Defaults to "chroma_db".
        chunk_size: Size of text chunks in characters. Must be between 100 and 10000.
                    Defaults to 1000.
        chunk_overlap: Number of overlapping characters between chunks. Must be >= 0.
                       Defaults to 100.
        model_embedding: Name of the Google Gemini embedding model to use.
                         Defaults to "gemini-embedding-001".
        model_llm: Name of the Google Gemini language model to use.
                   Defaults to "gemini-2.5-flash".
        search_type: Type of vector search to perform. Defaults to "similarity".
        search_kwargs: Dictionary of search parameters. Defaults to {"k": 3},
                       which retrieves the top 3 most similar chunks.
        temperature: Temperature parameter for LLM generation (0.0-1.0).
                     Lower values make output more deterministic. Defaults to 0.0.
    
    Example:
        >>> rag = RAGSystem(
        ...     file_path="document.pdf",
        ...     prompt_template="Context: {context}\\nQuestion: {question}"
        ... )
        >>> answer = rag.ask("What is the main topic?")
        >>> print(answer)
    """
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
        """
        Load and parse PDF document from the specified file path.
        
        This method uses PyPDFLoader to extract text content from a PDF file.
        Each page of the PDF is converted into a Document object containing
        the page content and metadata.
        
        Returns:
            list[Document]: A list of Document objects, where each Document
                           represents a page from the PDF. Each Document contains:
                           - page_content: The text content of the page
                           - metadata: Page number and source file information
        
        Raises:
            FileNotFoundError: If the specified file path does not exist.
            Exception: For other PDF parsing errors (corrupted file, unsupported format, etc.).
        
        Note:
            Returns None if an error occurs during loading. Consider raising
            exceptions instead for better error handling.
        """
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")
            
            logger.info(f"Loading PDF file from {self.file_path}")
            loader = PyPDFLoader(self.file_path)
            data = loader.load()
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def _split_data(self) -> list[Document]:
        """
        Split loaded documents into smaller text chunks.
        
        This method uses RecursiveCharacterTextSplitter to divide the document
        content into smaller chunks of specified size. Chunks overlap by a
        specified number of characters to maintain context continuity.
        
        The splitting strategy attempts to split on paragraph boundaries first,
        then sentences, then words, and finally characters to respect natural
        text boundaries while maintaining chunk size constraints.
        
        Returns:
            list[Document]: A list of Document objects, each representing a
                           text chunk. Each chunk has:
                           - page_content: The chunk text
                           - metadata: Original page number and source information
        
        Raises:
            Exception: If splitting fails (e.g., if _load_data() returns None).
        
        Note:
            Returns None if an error occurs. The chunk size and overlap are
            controlled by self.chunk_size and self.chunk_overlap attributes.
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            chunks = text_splitter.split_documents(self._load_data())
            logger.info(f"Splitted data into chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return None
    
    def _embed_data_google(self) -> GoogleGenerativeAIEmbeddings:
        """
        Initialize Google Generative AI embeddings model.
        
        This method creates and returns an embeddings model instance that can
        convert text into vector embeddings. The embeddings are used to create
        vector representations of document chunks for similarity search.
        
        The model is initialized using the Google Generative AI API. Requires
        GOOGLE_API_KEY environment variable to be set.
        
        Returns:
            GoogleGenerativeAIEmbeddings: An embeddings model instance that can
                                         convert text strings into vector embeddings.
        
        Raises:
            ValueError: If GOOGLE_API_KEY environment variable is not set.
            Exception: If model initialization fails (API errors, network issues, etc.).
        
        Note:
            Returns None if an error occurs. The model name is specified by
            self.model_embedding attribute (default: "gemini-embedding-001").
        """
        try:
            client = genai.Client()
            embeddings = GoogleGenerativeAIEmbeddings(model=self.model_embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding data: {e}")
            return None
    
    def _store_data(self) -> Chroma:
        """
        Store document chunks in ChromaDB vector database.
        
        This method manages the vector database lifecycle:
        - If the database exists, it loads the existing vectorstore
        - If the database doesn't exist, it creates a new one by:
          1. Splitting documents into chunks
          2. Generating embeddings for each chunk
          3. Storing embeddings in ChromaDB
        
        The database is persisted to disk in the directory specified by self.db_name,
        allowing the vectorstore to be reused across sessions without re-embedding.
        
        Returns:
            Chroma: A ChromaDB vectorstore instance that provides methods for
                   similarity search and retrieval of document chunks.
        
        Raises:
            Exception: If database operations fail (disk errors, embedding failures, etc.).
        
        Note:
            Returns None if an error occurs. The database persists automatically
            to the directory specified by self.db_name.
        """
        try:
            if os.path.exists(self.db_name):
                vectorstore = Chroma(
                    persist_directory=self.db_name,
                    embedding_function=self._embed_data_google()
                )
                logger.info(f"Loaded existing Chroma database from {self.db_name}")
            else:
                vectorstore = Chroma.from_documents(
                documents=self._split_data(), 
                embedding=self._embed_data_google(),
                persist_directory=self.db_name)
                logger.info(f"Created new Chroma database in {self.db_name}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            return None
        
    
    def _retriever(self) -> BaseRetriever:
        """
        Create a retriever for similarity search in the vector database.
        
        This method creates a retriever object that can search the vector database
        to find the most relevant document chunks for a given query. The retriever
        uses the configured search type and parameters.
        
        Returns:
            BaseRetriever: A retriever instance that provides a get_relevant_documents()
                          method to search for similar chunks based on query embeddings.
        
        Raises:
            Exception: If retriever creation fails (database not initialized, etc.).
        
        Note:
            Returns None if an error occurs. The search behavior is controlled by:
            - self.search_type: Type of search ("similarity", "mmr", etc.)
            - self.search_kwargs: Additional parameters (e.g., {"k": 3} for top-3 results)
        """
        try:
            retriever = self._store_data().as_retriever(
                search_type=self.search_type,
                search_kwargs=self.search_kwargs
            )
            logger.info(f"Retrieved data from Chroma")
            return retriever
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return None
    
    def _prompt_template(self) -> ChatPromptTemplate:
        """
        Create a prompt template from the template string.
        
        This method converts the prompt template string into a ChatPromptTemplate
        object that can be used with LangChain's LLM chain. The template should
        contain {context} and {question} placeholders that will be filled with
        retrieved context and user question respectively.
        
        Returns:
            ChatPromptTemplate: A prompt template object that can format prompts
                               with context and question variables.
        
        Raises:
            ValueError: If the template string is invalid or missing required placeholders.
        """
        prompt_template = ChatPromptTemplate.from_template(self.prompt_template)
        return prompt_template
    
    def _llm_google(self) -> ChatGoogleGenerativeAI:
        """
        Initialize Google Generative AI language model.
        
        This method creates and returns a language model instance that can generate
        text responses. The model is configured with the specified model name and
        temperature parameter.
        
        The temperature parameter controls the randomness of the output:
        - Lower values (0.0-0.3): More deterministic, focused responses
        - Higher values (0.7-1.0): More creative, varied responses
        
        Returns:
            ChatGoogleGenerativeAI: A language model instance that can generate
                                   text responses based on prompts.
        
        Raises:
            ValueError: If GOOGLE_API_KEY environment variable is not set.
            Exception: If model initialization fails (API errors, invalid model name, etc.).
        
        Note:
            Returns None if an error occurs. The model name and temperature are
            controlled by self.model_llm and self.temperature attributes.
        """
        try:
            llm = ChatGoogleGenerativeAI(model=self.model_llm, temperature=self.temperature)
            return llm
        except Exception as e:
            logger.error(f"Error LLM: {e}")
            return None
    
    @staticmethod
    def format_docs(docs: list[Document]) -> str:
        """
        Format a list of Document objects into a single string.
        
        This static method concatenates the page content from multiple Document
        objects into a single string, separated by double newlines. This formatted
        string is typically used as context in the LLM prompt.
        
        Args:
            docs: A list of Document objects to format. Each Document should have
                  a page_content attribute containing the text content.
        
        Returns:
            str: A single string containing all document contents, separated by
                 double newlines ("\\n\\n").
        
        Example:
            >>> docs = [Document(page_content="Chunk 1"), Document(page_content="Chunk 2")]
            >>> formatted = RAGSystem.format_docs(docs)
            >>> print(formatted)
            "Chunk 1\\n\\nChunk 2"
        """
        return "\n\n".join(doc.page_content for doc in docs)

    def _chain(self) -> Runnable:
        """
        Create the RAG processing chain.
        
        This method constructs a LangChain Runnable chain that orchestrates the
        entire RAG pipeline:
        1. Takes a question as input
        2. Retrieves relevant document chunks using the retriever
        3. Formats the chunks into a context string
        4. Formats the prompt with context and question
        5. Generates an answer using the LLM
        6. Parses the output as a string
        
        The chain uses LangChain's pipe operator (|) to connect components in
        a functional programming style.
        
        Returns:
            Runnable: A LangChain Runnable object that can be invoked with a
                     dictionary containing a "question" key. The chain will return
                     a string answer.
        
        Raises:
            Exception: If chain construction fails (missing components, etc.).
        
        Note:
            Returns None if an error occurs. The chain combines:
            - Retriever: Finds relevant chunks
            - Prompt template: Formats the prompt
            - LLM: Generates the answer
            - Output parser: Converts to string
        """
        try:
            chain = (
            {"context": RunnableLambda(lambda x: x["question"]) | self._retriever() | self.format_docs
            , "question": RunnablePassthrough()} 
            | self._prompt_template() 
            | self._llm_google() 
            | StrOutputParser()
            )
            logger.info(f"Chain created")
            return chain
        except Exception as e:
            logger.error(f"Error chain: {e}")
            return None

    def ask(self, question: str) -> str:
        """
        Process a user question and return an answer using the RAG system.
        
        This is the main public method that orchestrates the entire RAG pipeline
        to answer a user's question. It:
        1. Takes a question string as input
        2. Executes the RAG chain (retrieval + generation)
        3. Logs the question and answer
        4. Returns the generated answer
        
        Args:
            question: The user's question to answer. Should be a clear, specific
                     question about the content in the loaded PDF document.
        
        Returns:
            str: The generated answer based on the retrieved context from the
                 document. Returns None if an error occurs during processing.
        
        Raises:
            Exception: If the RAG chain execution fails (retrieval errors, LLM
                      API errors, etc.).
        
        Note:
            The method logs both the question and answer at INFO level for
            debugging and monitoring purposes.
        """
        try:
            result = self._chain().invoke({"question": question})
            logger.info("-" * 50)
            logger.info(f"Question: {question}")
            logger.info(f"Answer: {result}")
        except Exception as e:
            logger.error(f"Error: {e}")
            return None
        return result
