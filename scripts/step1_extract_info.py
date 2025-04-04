from __future__ import annotations # type checking
from pathlib import Path
from pydantic import BaseModel

import os, random, argparse, asyncio, logging
import typing as t, pandas as pd
from tqdm import tqdm

from ragas.prompt import PydanticPrompt

from ragpas.llm import get_llm


logger = logging.getLogger(__name__)


class ExtractionInput(BaseModel):
    context: t.Optional[str]
    target: t.Optional[str]

class ExtractionOutput(BaseModel):
    all_info: t.Optional[dict[str, t.Any]]

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
        ),
        (
            ExtractionInput(
                context="Patient: hye, My aunt is having shortness of breath and she is on vent now. Her breast showed some kind of infection that apparently turned out to have black blisters. I could she those spreading. When we went for lungs x-ray doctor said she has accumulation of water in her lungs Doctor: Thanks for your question on Chat Doctor. I can understand your aunts situation and problem. By your history and description, possibility of bacterial infection especially staphylococcus is more in her case. She is having pleural effusion and infective skin lesions on breast. Staphylococcus can cause pleural effusion and blister formation on skin. So chances of staphylococcal infection is more in her case. Better to send pleural fluid culture and sensitivity for the diagnosis of staphylococcal infection. This will also tell about effective antibiotic therapy. With appropriate antibiotics and Care, this infection can be treated. Hope I have solved your query. Wishing good health to your aunt. Thanks.",
                target="Person"
            ),
            ExtractionOutput(
                all_info={
                    'Symptoms': 'Shortness of breath, accumulation of water in lungs, infective skin lesions on breast with black blisters', 
                    'Diagnosis': 'Possibility of staphylococcal infection, pleural effusion', 
                    'Suggestion': 'Send pleural fluid culture and sensitivity for diagnosis and effective antibiotic therapy'
                }
            )
        )
    ]


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
    df = pd.read_csv(file_path)
    contexts = df["context"].tolist()
    if "target" in df.columns:
        targets = df["target"].tolist()
        return contexts, targets
    else:
        return contexts, None
    
def devide_info(all_info: dict[str, t.Any]):
    privacy_info = {}
    known_info = {}
    if all_info:
        keys = list(all_info.keys())
        num_privacy_attributes = len(keys) // 2
        privacy_keys = random.sample(keys, num_privacy_attributes)
        for key in keys:
            if key in privacy_keys:
                privacy_info[key] = all_info[key]
            else:
                known_info[key] = all_info[key]
    return privacy_info, known_info

def main():
    argparser = argparse.ArgumentParser(description="extract all information from a context")
    argparser.add_argument("-m", "--model", type=str, default="doubao-1-5-lite", help="model name")
    argparser.add_argument("-i", "--input_file_path", type=str, required=True, help="input file path")
    argparser.add_argument("-o" ,"--output_file_path", type=str, required=True, help="output file path")
    argparser.add_argument("-t", "--target", type=str, default="Person", help="target for extraction")
    args = argparser.parse_args()
    
    # check input file path is a directory
    if os.path.isdir(args.input_file_path):
        input_file_path = os.path.join(args.input_file_path, "input.csv")
        if not os.path.exists(input_file_path):
            logger.error(f"Input file not found: {input_file_path}")
            return
        args.input_file_path = input_file_path
    elif not os.path.exists(args.input_file_path):
        logger.error(f"Input file not found: {args.input_file_path}")
        return
    
    if os.path.isdir(args.output_file_path):
        output_file_path = os.path.join(args.output_file_path, "info.csv")
        args.output_file_path = output_file_path
    if os.path.exists(args.output_file_path):
        logger.error(f"Output file already exists: {args.output_file_path}")
        return
    else:
        pd.DataFrame(columns=["original_context", "all_info", "privacy_info", "known_info", "target"]).to_csv(args.output_file_path, index=False)

    # read contexts and target from input file
    contexts, targets = read_contexts_from_csv(args.input_file_path)
    
    batch_size = 10
    batch_original_contexts = []
    batch_all_info = []
    batch_privacy_info = []
    batch_known_info = []
    batch_targets = []

    for i in tqdm(range(len(contexts)), desc="Processing contexts"):
        context = str(contexts[i])
        if targets is not None:
            target = targets[i]
        else:
            target = args.target
            
        all_info = extract_info_from_context(context, target, args.model)
        privacy_info, known_info = devide_info(all_info)
        batch_original_contexts.append(context)
        batch_all_info.append(all_info)
        batch_privacy_info.append(privacy_info)
        batch_known_info.append(known_info)
        batch_targets.append(target)

        if (i + 1) % batch_size == 0 or i == len(contexts) - 1:
            batch_df = pd.DataFrame({
                "original_context": batch_original_contexts,
                "all_info": batch_all_info,
                "privacy_info": batch_privacy_info,
                "known_info": batch_known_info,
                "target": batch_targets
            })
            batch_df.to_csv(args.output_file_path, mode='a', header=False, index=False)

            batch_original_contexts = []
            batch_all_info = []
            batch_privacy_info = []
            batch_known_info = []
            batch_targets = []

            logger.info(f"Processed and saved {i + 1} contexts")

    logger.info(f"Extraction completed. Results saved to {args.output_file_path}")

if __name__ == "__main__":
    from dotenv import load_dotenv

    root_dir = str(Path().absolute().parent)
    env_path = f"{root_dir}/.env"
    load_dotenv(env_path)

    main()
