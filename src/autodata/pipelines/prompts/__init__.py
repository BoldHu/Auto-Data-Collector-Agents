"""Pipeline prompts package."""

from src.autodata.pipelines.prompts.text_cleaning_prompts import (
    ZH_CLEANING_PROMPT,
    EN_CLEANING_PROMPT,
    KNOWLEDGE_EXTRACTION_PROMPT,
    SFT_GENERATION_PROMPT,
    QUALITY_VERIFICATION_PROMPT,
    PROMPT_VERSION,
    get_cleaning_prompt,
    get_knowledge_extraction_prompt,
    get_sft_generation_prompt,
    get_quality_verification_prompt,
)

__all__ = [
    "ZH_CLEANING_PROMPT",
    "EN_CLEANING_PROMPT",
    "KNOWLEDGE_EXTRACTION_PROMPT",
    "SFT_GENERATION_PROMPT",
    "QUALITY_VERIFICATION_PROMPT",
    "PROMPT_VERSION",
    "get_cleaning_prompt",
    "get_knowledge_extraction_prompt",
    "get_sft_generation_prompt",
    "get_quality_verification_prompt",
]