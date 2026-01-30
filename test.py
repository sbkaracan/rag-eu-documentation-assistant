from rag import RAGSystem

#rag = RAGSystem(file_path="a_short_guide_to_eu.pdf", 
#    prompt_template="""You are a virtual assistant for the European Union.
#    Your goal is to assist users who wants to know about the European Union.
#    You must use the relevant documentation given to you to answer user queries.
#    You can only answer questions about the European Union. 
#
#    Context: {context}
#    Question: {question}""")


#result = rag.ask("What is the EU doing to improve things where I live?")

#result2 = rag.ask("What is the EU doing?")

rag = RAGSystem(file_path=["test_docs/test_doc_1.pdf", "test_docs/test_doc_2.pdf"], prompt_template="""
You are a virtual assistant for the academic papers.
Your goal is to assist users who wants to know about the papers you have read.
You must use the relevant documentation given to you to answer user queries.
You can only answer questions about the paper review process.
Context: {context}
Question: {question}""")

result = rag.ask("""What are the contributors of the 
'Evaluating the Effectiveness of GPT-4 Turbo in Creating Defeaters for Assurance Cases' and 
'Position paper: a vision for the dynamic safety assurance of ML-enabled autonomous driving systems' ?""")