import chainlit as cl
from llm import load_model,create_coder_template
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain


@cl.on_chat_start
async def on_chat_start():
    
    model_path="mistral-7b-instruct-v0.1.Q6_K.gguf"
    menager=CallbackManager(StdOutCallbackHandler())
    model=load_model(model_path=model_path,callback_menager=None)
    
    prompt = create_coder_template()
    
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"problem_statement": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
    
     
