import asyncio
from ragpas.llm import get_llm, get_embeddings
from ragas import SingleTurnSample
from ragas.metrics import ContextRecall, ResponseRelevancy, SemanticSimilarity, ExactMatch, RougeScore, BleuScore

# Context Recall
async def acalculateContextRecall(user_input: str, retrieved_contexts: list[str], response: str, reference: str):
    evaluator_llm = get_llm("gpt-4o-mini")
    metric = ContextRecall(llm=evaluator_llm)
    sample = SingleTurnSample(user_input=user_input, retrieved_contexts=retrieved_contexts, response=response, reference=reference)
    score = await metric.single_turn_ascore(sample)
    return score

def calculateContextRecall(user_input: str, retrieved_contexts: list[str], response: str, reference: str):
    return asyncio.run(acalculateContextRecall(user_input=user_input, retrieved_contexts=retrieved_contexts, response=response, reference=reference))


# Response Relevancy
async def acalculateResponseRelevancy(user_input: str, retrieved_contexts: list[str], response: str):
    evaluator_llm = get_llm("gpt-4o-mini")
    evaluator_embeddings = get_embeddings("text-embedding")
    metric = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    sample = SingleTurnSample(user_input=user_input, retrieved_contexts=retrieved_contexts, response=response)
    score = await metric.single_turn_ascore(sample)
    return score

def calculateResponseRelevancy(user_input: str, retrieved_contexts: list[str], response: str):
    return asyncio.run(acalculateResponseRelevancy(user_input=user_input, retrieved_contexts=retrieved_contexts, response=response))


# Semantic Similarity
async def acalculateSemanticSimilarity(response: str, reference: str):
    evaluator_embeddings = get_embeddings("text-embedding")
    metric = SemanticSimilarity(embeddings=evaluator_embeddings)
    sample = SingleTurnSample(response=response, reference=reference)
    score = await metric.single_turn_ascore(sample)
    return score

def calculateSemanticSimilarity(response: str, reference: str):
    return asyncio.run(acalculateSemanticSimilarity(response=response, reference=reference))


# Exact Match
async def acalculateExactMatch(response: str, reference: str):
    metric = ExactMatch()
    sample = SingleTurnSample(response=response, reference=reference)
    score = await metric.single_turn_ascore(sample)
    return score

def calculateExactMatch(response: str, reference: str):
    return asyncio.run(acalculateExactMatch(response=response, reference=reference))


# Rouge Score
async def acalculateRougeScore(response: str, reference: str):
    # default rouge type is rouge-l
    metric = RougeScore(rouge_type="rougeL")
    sample = SingleTurnSample(response=response, reference=reference)
    score = await metric.single_turn_ascore(sample)
    return score

def calculateRougeScore(response: str, reference: str):
    return asyncio.run(acalculateRougeScore(response=response, reference=reference))


# Bleu Score
async def acalculateBleuScore(response: str, reference: str):
    metric = BleuScore()
    sample = SingleTurnSample(response=response, reference=reference)
    score = await metric.single_turn_ascore(sample)
    return score

def calculateBleuScore(response: str, reference: str):
    return asyncio.run(acalculateBleuScore(response=response, reference=reference))


# function class
class performance:
    ContextRecall = calculateContextRecall
    ResponseRelevancy = calculateResponseRelevancy
    SemanticSimilarity = calculateSemanticSimilarity
    EM = calculateExactMatch
    Rouge = calculateRougeScore
    Bleu = calculateBleuScore

class aperformance:
    ContextRecall = acalculateContextRecall
    ResponseRelevancy = acalculateResponseRelevancy
    SemanticSimilarity = acalculateSemanticSimilarity
    EM = acalculateExactMatch
    Rouge = acalculateRougeScore
    Bleu = acalculateBleuScore