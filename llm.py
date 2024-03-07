from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager



def load_model(model_path,callback_menager):
   
    model =LlamaCpp(model_path=model_path,
                callback_manager=callback_menager,
                verbose=True,
                temperature=0.4,
                n_ctx=4096,
                max_tokens=4000,
                n_gpu_layers=33,
                n_batch=2064
                   )
    
    return(model)


def create_coder_template():
    template= """
    context: '''You are a skilled and experienced coder working for a reputable software development company. /
      Your role involves developing innovative solutions for various clients, ensuring the highest quality of code, /
      and collaborating effectively with your team members./
    '''

    prompt: In this context, please provide a well-structured and optimized code snippet for the following problem/
    {problem_statement}

    """
    prompt=PromptTemplate(template=template,input_variables=["problem_statement"])
    return(prompt)
    
    
    
    
def create_LLMchain(model,memory,prompt):

    llm_chain=LLMChain(llm=model,
                       memory=memory,
                       verbose=True,
                       prompt=prompt
                       )
    return(llm_chain)

def get_completion(llm_chain,question):
    return(llm_chain.run(question))


def main():
    # declaring variables
    model_path=r"C:\Users\Iheb\Desktop\projects\S_project\mistral-7b-instruct-v0.1.Q6_K.gguf"
    stream=StreamingStdOutCallbackHandler()
    memory=ConversationBufferMemory()
    callback_menager= CallbackManager([stream])
    
    # loading the local_model
    model=load_model(model_path=model_path,callback_menager=callback_menager)

    # create the prompt 
    prompt=create_coder_template()

    # initiate the llm-chain
    llm_chain=create_LLMchain(model=model,prompt=prompt,memory=memory)

    # get llm completion
    tryagain=False
    
    
        
    question=input("Hi, any code I can help you with today? \n")
    llm_chain.run(question)
       
        
        

          
        
     

if __name__=="__main__":
    main()



    