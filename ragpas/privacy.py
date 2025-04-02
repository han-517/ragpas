import os
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from metrics._attack_extraction import AttackExtraction, ComputeRubric

import asyncio

from metrics._target_identification import TargetIdentifier
from ragpas.llm import get_llm, get_embeddings
from ragpas.dataset_schema import PrivacySingleTurnSample


async def acalculateAttackExtraction(response: str, privacy_info: dict | str, target: str, feedback: bool = False, model: str = "doubao-1-5-lite"):
    evaluator_llm = get_llm(model=model)
    metric = AttackExtraction(llm=evaluator_llm, score_rubric=ComputeRubric.MAX, feedback=feedback)
    sample = PrivacySingleTurnSample(response=response, privacy_info=privacy_info, target=target)
    if feedback:
        score, feedback = await metric.single_turn_ascore(sample)
        return score, feedback
    score = await metric.single_turn_ascore(sample)
    return score

def calculateAttackExtraction(response: str, privacy_info: str, target: str, feedback: bool = False, model: str = "doubao-1-5-lite"):
    return asyncio.run(acalculateAttackExtraction(response=response, privacy_info=privacy_info, target=target, feedback=feedback, model=model))


async def acalcuateTargeIdentification(response: str, known_info: str, collection_name: str, original_context: str, model: str = "doubao-1-5-lite", target: str = "Person", embedding_name: str = "doubao-embedding", top_k: int = 5):
    # response:从RAG获取的未知信息
    # known_info:部分已知信息
    # collection_name:集合名
    root_dir = root_dir = str(Path().absolute().parent)
    evaluator_llm = get_llm(model=model)
    embedding = OpenAIEmbeddings(
     model=os.environ.get("EMBEDDING_MODEL"),
     api_key=os.environ.get("EMBEDDING_API_KEY"),
     base_url=os.environ.get("EMBEDDING_API_URL"),
     dimensions=os.environ.get("EMBEDDING_DIMENSIONS", None),
     check_embedding_ctx_length=False,
     openai_proxy="http://127.0.0.1:7897",
    )
    vectordb = Chroma(
        collection_name="retrieval_database",
        persist_directory=f"{root_dir}/data/database/chroma-doubao-2000/",
        embedding_function=embedding,
    )
    metric = TargetIdentifier(llm=evaluator_llm, vectordb=vectordb, llm_name=model, top_k=top_k)
    sample = PrivacySingleTurnSample(response=response, known_info=known_info, target=target, extension_fields={"original_context": original_context})
    score = await metric.single_turn_ascore(sample)
    return score


def calculateTargetIdentification(response: str, known_info: str, collection_name: str, original_context:str, model: str = "doubao-1-5-lite", target: str = None, embedding_name: str = "doubao-embedding", top_k: int = 5):
    return asyncio.run(acalcuateTargeIdentification(response=response, known_info=known_info, collection_name=collection_name, original_context=original_context, model=model, target=target, embedding_name=embedding_name, top_k=top_k))