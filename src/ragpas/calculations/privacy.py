from ragpas.calculations import AttackExtraction, ComputeRubric
from ragpas.utils.llm import get_llm
from ragpas.schemas.calculate import PrivacySingleTurnSample
import asyncio


async def acalculateAttackExtraction(response: str, privacy_info: dict | str, target: str, model: str, feedback: bool = False):
    evaluator_llm = get_llm(model=model)
    metric = AttackExtraction(llm=evaluator_llm, score_rubric=ComputeRubric.MAX, feedback=feedback)
    sample = PrivacySingleTurnSample(response=response, privacy_info=privacy_info, target=target)
    if feedback:
        score, feedback = await metric.single_turn_ascore(sample)
        return score, feedback
    score = await metric.single_turn_ascore(sample)
    return score


def calculateAttackExtraction(response: str, privacy_info: str, target: str, model: str, feedback: bool = False):
    return asyncio.run(acalculateAttackExtraction(response=response, privacy_info=privacy_info, target=target, model=model, feedback=feedback))