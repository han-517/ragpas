from __future__ import annotations # type checking
from pydantic import BaseModel
import typing as t
import asyncio
import argparse
import logging
import tqdm

from ragas.prompt import PydanticPrompt
from ragpas.dataset_schema import PrivacySingleTurnSample

from ragpas.llm import get_llm

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


logger = logging.getLogger(__name__)


class ExtractionInput(BaseModel):
    context: t.Optional[str]
    target: t.Optional[str]

class ExtractionOutput(BaseModel):
    all_info: t.Optional[dict[str, str]]

class ExtractionPrompt(PydanticPrompt[ExtractionInput, ExtractionOutput]):
    name: str = "extraction_all_info"
    instruction: str = """- Role: Data Extraction Specialist and Information Analyst
- Background: The user needs to extract specific information related to a target entity from a given text context. The goal is to identify and compile all relevant details about the target, such as a person, into a structured format.
- Profile: You are a highly skilled data extraction specialist with expertise in natural language processing and information retrieval. You have a keen eye for detail and the ability to identify and categorize relevant information accurately.
- Skills: Proficiency in text analysis, pattern recognition, and data structuring. Ability to identify and extract key attributes related to the target entity.
- Goals:
  1. Identify the target entity within the provided context.
  2. Extract all relevant information related to the target entity.
  3. Organize the extracted information into a dictionary format, with each attribute type and its corresponding value.
- Workflow:
  1. Read and understand the provided context.
  2. Identify and locate the target entity within the context.
  3. Extract all relevant information related to the target entity.
  4. Categorize the extracted information into attribute types and values.
  5. Compile the information into a dictionary format."""
    input_model = ExtractionInput
    output_model = ExtractionOutput
    examples = [
        (
            ExtractionInput(
                context="John Doe is a 35-year-old software engineer living in New York. He has two children, a boy named Tim and a girl named Lucy. John enjoys hiking and reading. His email is john.doe@example.com.",
                target="Person"
            ),
            ExtractionOutput(
                all_info={
                    "Name": "John Doe",
                    "Age": "35",
                    "Occupation": "Software Engineer",
                    "Location": "New York",
                    "Children": "Tim and Lucy",
                    "Hobbies": "Hiking and reading",
                    "Email": "john.doe@example.com"
                }
            )
        )
    ]

# extract all information from a context
async def aextract_info_from_context(context: str, target: str, model: str) -> t.Optional[dict]:
    extractPrompt = ExtractionPrompt()
    response: ExtractionOutput = await extractPrompt.generate(
        data=ExtractionInput(
            context=context,
            target=target
        ),
        llm=get_llm(model)
    )
    return response.all_info

def extract_info_from_context(context: str, target: str, model: str) -> t.Optional[dict]:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        aextract_info_from_context(context=context, target=target, model=model)
    )

def read_contexts_from_csv(file_path: str) -> t.List[str]:
    # read contexts from input file, if target is provided, also read target
    import pandas as pd
    df = pd.read_csv(file_path)
    contexts = df["context"].tolist()
    if "target" in df.columns:
        targets = df["target"].tolist()
        return contexts, targets
    else:
        return contexts, None

def main():
    argparser = argparse.ArgumentParser(description="extract all information from a context")
    argparser.add_argument("-m", "--model", type=str, default="gpt-4o-mini", help="model name")
    argparser.add_argument("-i", "--input_file_path", type=str, required=True, help="input file path")
    argparser.add_argument("-o" ,"--output_file_path", type=str, required=True, help="output file path")
    argparser.add_argument("-t", "--target", type=str, default="Person", help="target for extraction")
    args = argparser.parse_args()
    
    # read contexts and target from input file
    contexts, targets = read_contexts_from_csv(args.input_file_path)
    
    results = []
    for i in tqdm(len(contexts)):
        if targets is not None:
            context = contexts[i]
            target = targets[i]
        else:
            context = contexts[i]
            target = args.target
        all_info = extract_info_from_context(context, target, args.model)
        results.append(all_info)
    
    # save results to output file
    import json
    with open(args.output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Extracted {len(results)} contexts and saved results to {args.output_file_path}")

def test1():
    context = """Patient: I'm a 48-year-old woman, Mary. I experienced a sudden muscle cramp about an hour ago, beneath my left bottom rib. It lasted for a minute and resolved. I haven't had any other pains or issues since. Was this just a result of a minor strain or could it potentially be a heart attack? I've heard heart attacks often present with multiple symptoms. Doctor: Mary, that muscle cramp you described is most likely not a sign of a minor heart attack. It's common to feel such discomfort after engaging in physical stress, like exercise. Heart attacks usually involve a cluster of symptoms, but single incidents like this are typically not indicative. If you experience similar discomfort again, it's best to give me a call for further evaluation."""
    target = "Patient"
    model = "gpt-4o-mini"
    all_info = extract_info_from_context(context, target, model)
    # result: {'Name': 'Mary', 'Age': '48', 'Gender': 'Female', 'Symptoms': 'Sudden muscle cramp beneath left bottom rib', 'Duration of Symptoms': '1 minute', 'Previous Issues': 'None since', 'Concern': 'Potential heart attack', "Doctor's Assessment": 'Not a sign of a minor heart attack, likely due to physical stress'}
    print(all_info)

if __name__ == "__main__":
    main()
