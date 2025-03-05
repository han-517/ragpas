from metrics._attack_extraction import AttackExtraction, ComputeRubric

import asyncio
from ragpas.llm import get_llm
from ragpas.dataset_schema import PrivacySingleTurnSample


async def acalculateAttackExtraction(response: str, privacy_info: dict | str, target: str):
    evaluator_llm = get_llm("doubao-1-5-lite")
    metric = AttackExtraction(llm=evaluator_llm, score_rubric=ComputeRubric.MAX)
    sample = PrivacySingleTurnSample(response=response, privacy_info=privacy_info, target=target)
    score = await metric.single_turn_ascore(sample)
    return score

def calculateAttackExtraction(response: str, privacy_info: str, target: str):
    return asyncio.run(acalculateAttackExtraction(response=response, privacy_info=privacy_info, target=target))