# RAG System with LangChain

RAG stands for retrieval augmented generation, a technique developped by researchers to eliminate the need for retraining the model's parameters and reduce model hallucinations.
The main concept of RAG is 
1-.dividing the source file into chunks 
2.Encode the chunks using sentence-encoder
3.instead of the prompt going straight to the model, we mesure the similarites between the encoded prompt and encoded chunks and retain the chunks with the hightest score(similarity metrics such as cosine similarity)
4.Injecting the prompt with the additional context and deliver it to the model for inference.
It allows the LLM to access some additional context besides that of its training for a more efficient response  
Implemented using python langchain, GGUF LLM that can be found on HuggingFace and chainlit for the UI. 

## Installation

1. Clone the repository.
2. Install the necessary dependencies.
3. Run the application

(Have fun trying it out)
