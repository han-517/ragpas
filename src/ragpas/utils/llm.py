import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragpas.config import get_global_config

global_config = get_global_config()

def get_llm(model: str) -> LangchainLLMWrapper:
    return LangchainLLMWrapper(ChatOpenAI(
        model=model,
        api_key=os.environ.get('OPENAI_API_KEY'),
        base_url=os.environ.get('OPENAI_API_URL'),
        openai_proxy=global_config.proxy
    ))

def get_embeddings(model: str) -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        model=model,
        api_key=os.environ.get('OPENAI_API_KEY'),
        base_url=os.environ.get('OPENAI_API_URL'),
        openai_proxy=global_config.proxy
    ))