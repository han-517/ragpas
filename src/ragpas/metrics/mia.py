from __future__ import annotations # type checking

import os, random, asyncio, logging
import typing as t, pandas as pd

from tqdm import tqdm
from ragpas.utils.llm import get_llm
from ragpas.config import get_mia_config, update_config, get_rag_config
from .prompt import ExtractionInput, ExtractionOutput, ExtractionPrompt, GenerationInput, GenerationOutput, GenerationPrompt
from ragpas.calculations import calculateAttackExtraction
from ragpas.rag import NaiveRAG, RAGDocument, BaseRAG



logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def aextract_info_from_context(context: str, model: str, target: str) -> t.Optional[dict]:
    extractPrompt = ExtractionPrompt()
    print(f"Extracting information from context with model: {model}")
    response: ExtractionOutput = await extractPrompt.generate(
        data=ExtractionInput(
            context=context,
            target=target
        ),
        llm=get_llm(model)
    )
    return response.all_info

def extract_info_from_context(context: str, model: str, target: str) -> t.Optional[dict]:
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
    
#TODO: Add other methods to devide information into privacy and known info
# Currently, it randomly selects half of the attributes as privacy info
# and the rest as known info.
# This is a placeholder and should be replaced with a more sophisticated method.
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

async def agenerate_attack_prompt(privacy_info_types: list[str], known_info: dict[str, str | list[str]], target: str, model: str) -> str:
    evaluator_llm = get_llm(model=model)
    prompt = GenerationPrompt()
    response: GenerationOutput = await prompt.generate(
        data=GenerationInput(
            privacy_info_types=privacy_info_types,
            known_info=known_info,
            target=target
        ),
        llm=evaluator_llm
    )
    return response.attack_prompt

def generate_attack_prompt(privacy_info_types: list[str], known_info: dict[str, str | list[str]], target: str, model: str) -> str:
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        agenerate_attack_prompt(privacy_info_types=privacy_info_types, known_info=known_info, target=target, model=model)
    )

#TODO: Add other data formats like JSON, Database, etc.
# Currently, it only supports CSV format.
def initialize_mia_output_file() -> None:
    """Initialize the output CSV file with headers."""
    output_path = get_mia_config().mia_output_path
    pd.DataFrame(columns=["all_info", "privacy_info", "known_info", "target", "attack_prompt", "response"]).to_csv(
        output_path, index=False
    )

def save_mia_output_file(all_info: list, privacy_info: list, known_info: list, target: list, attack_prompt: list, response: list) -> None:
    """Save the output file to the specified path."""
    output_path = get_mia_config().mia_output_path
    df = pd.DataFrame({
        "all_info": all_info,
        "privacy_info": privacy_info,
        "known_info": known_info,
        "target": target,
        "attack_prompt": attack_prompt,
        "response": response
    })
    df.to_csv(output_path, mode='a', header=False, index=False)

def initialize_calculation_output_file() -> None:
    """Initialize the calculation output file with headers."""
    output_path = get_mia_config().calculation_output_path
    pd.DataFrame(columns=["score", "feedback"]).to_csv(
        output_path, index=False
    )

def save_calculation_output_file(scores: list, feedback: list) -> None:
    """Save the calculation output file to the specified path."""
    output_path = get_mia_config().calculation_output_path
    df = pd.DataFrame({
        "score": scores,
        "feedback": feedback
    })
    df.to_csv(output_path, mode='a', header=False, index=False)

def process_contexts_batch(
    contexts: t.List[str],
    targets: t.Optional[t.List[str]] = None,
    rag: BaseRAG = None
) -> None:
    """Process contexts in batches and save results to CSV file using global config."""
    config = get_mia_config()
    
    batch_all_info = []
    batch_privacy_info = []
    batch_known_info = []
    batch_targets = []
    batch_attack_prompts = []
    batch_responses = []
    batch_scores = []
    batch_feedback = []

    for i in tqdm(range(len(contexts)), desc="[MIA] Processing contexts"):
        context = str(contexts[i])
        if targets is not None and targets[i].strip() != "":
            target = targets[i]
        else:
            target = config.target
            
        logger.info("Extracting information from context...")
        all_info = extract_info_from_context(context=context, target=target, model=config.extract_model)

        logger.info("Deviding information into privacy and known info...")
        privacy_info, known_info = devide_info(all_info)

        logger.info("Generating attack prompt...")
        attack_prompt = generate_attack_prompt(
            privacy_info_types=list(privacy_info.keys()),
            known_info=known_info,
            target=target,
            model=config.generate_model
        )

        response = rag.generate_response(attack_prompt)

        score, feedback = calculateAttackExtraction(
            response=response,
            privacy_info=privacy_info,
            target=target,
            model=config.evaluate_model,
            feedback=True
        )

        batch_scores.append(score)
        batch_feedback.append(feedback)

        batch_all_info.append(all_info)
        batch_privacy_info.append(privacy_info)
        batch_known_info.append(known_info)
        batch_targets.append(target)
        batch_attack_prompts.append(attack_prompt)
        batch_responses.append(response)


        if (i + 1) % config.save_step == 0 or i == len(contexts) - 1:
            save_mia_output_file(
                all_info=batch_all_info,
                privacy_info=batch_privacy_info,
                known_info=batch_known_info,
                target=batch_targets,
                attack_prompt=batch_attack_prompts,
                response=batch_responses
            )

            batch_all_info = []
            batch_privacy_info = []
            batch_known_info = []
            batch_targets = []
            batch_attack_prompts = []
            batch_responses = []

            save_calculation_output_file(
                scores=batch_scores,
                feedback=batch_feedback
            )

            batch_scores = []
            batch_feedback = []

            logger.info(f"Processed {i + 1} contexts, saving...")

    logger.info(f"Membership inference attack completed. Results saved to {config.mia_output_filename}")

def run_mia(
    extract_model: str = None,
    generate_model: str = None,
    evaluate_model: str = None,
    dataset_input_path: str = None,
    mia_output_path: str = None,
    calculation_output_path: str = None,
    target: str = None,
    save_step: int = None,
    save_config: bool = False,
) -> None:
    """Run MIA extraction process using global configuration.
    """
    # Load or get global configuration
    config = get_mia_config()
    
    # Update config with provided parameters (don't save automatically)
    config_updates = {}
    if extract_model:
        config_updates['extract_model'] = extract_model
    if generate_model:
        config_updates['generate_model'] = generate_model
    if evaluate_model:
        config_updates['evaluate_model'] = evaluate_model
    if dataset_input_path:
        config_updates['dataset_input_path'] = dataset_input_path
    if mia_output_path:
        config_updates['mia_output_path'] = mia_output_path
    if calculation_output_path:
        config_updates['calculation_output_path'] = calculation_output_path
    if target:
        config_updates['target'] = target
    if save_step:
        config_updates['save_step'] = save_step
    
    # If there are updates, apply them to the config
    # and optionally save to the config file
    if config_updates:
        update_config("mia", save=save_config, **config_updates)
    
    setup_logging(config.log_level)
    
    initialize_mia_output_file()
    initialize_calculation_output_file()

    contexts, targets = read_contexts_from_csv(config.dataset_input_path)

    rag = NaiveRAG(get_rag_config())
    rag.initialize()

    logger.info("Adding contexts to RAG database...")
    rag.add_documents(contexts)
    logger.info(f"{len(contexts)} contexts have been added to RAG database.")

    process_contexts_batch(contexts, targets, rag=rag)
