from ragas.dataset_schema import BaseSample
import typing as t


class PrivacySingleTurnSample(BaseSample):
    user_input: t.Optional[str] = None
    retrieved_contexts: t.Optional[t.List[str]] = None
    response: t.Optional[str] = None
    reference: t.Optional[str] = None
    privacy_info: t.Optional[dict] | t.Optional[str] = None
    purpose: t.Optional[str] = None
    rubrics: t.Optional[t.Dict[str, str]] = None
