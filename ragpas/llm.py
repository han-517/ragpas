import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def get_llm(model: str) -> LangchainLLMWrapper:
    # check if the model is supported
    model_dict = {
        'gpt-4o': 'gpt-4o',
        'gpt-4o-mini': 'gpt-4o-mini',
        'deepseek-reasoner': 'deepseek-ai/deepseek-r1',
        'doubao-1-5-lite': 'doubao-1-5-lite-32k-250115',
        'doubao-1-5-pro': 'doubao-1-5-pro-32k-250115'
    }
    if model not in model_dict:
        raise ValueError(f"Model {model} is not supported. Supported models are {model_dict.keys()}")
    return LangchainLLMWrapper(ChatOpenAI(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model=model_dict[model],
        base_url=os.environ.get('OPENAI_API_URL'),
        openai_proxy="http://127.0.0.1:7890"
    ))

def get_embeddings(model: str) -> LangchainEmbeddingsWrapper:
    # check if the model is supported
    model_dict = {
        'text-embedding': 'text-embedding-3-large',
    }
    if model not in model_dict:
        raise ValueError(f"Model {model} is not supported. Supported models are {model_dict.keys()}")
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model=model_dict[model],
        base_url=os.environ.get('OPENAI_API_URL'),
        openai_proxy="http://127.0.0.1:7890"
    ))

if __name__ == '__main__':
    # load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    llm = ChatOpenAI(
            model='deepseek-ai/deepseek-r1',
            api_key=os.environ.get('OPENAI_API_KEY'),
            base_url=os.environ.get('OPENAI_API_URL'),
            openai_proxy="http://127.0.0.1:7890"
        )
    
    system_template = "Translate the following from English into {language}"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

    response = llm.invoke(prompt)
    print(response.content)