from typing import List, Optional

from pydantic import BaseModel, Field


# --- Stage 1: Decomposition ---
class DecomposedParameter(BaseModel):
    core_objective: str = Field(
        description="The primary goal of the audit parameter. What exactly are we checking?"
    )
    key_entities: List[str] = Field(
        description="Important nouns or concepts (e.g., 'Terms and Conditions', 'Customer', 'Consent')."
    )
    implied_rules: List[str] = Field(
        description="Rules or procedures that must be verified in the client's knowledge base to evaluate this parameter."
    )
    initial_search_concepts: List[str] = Field(
        description="Broad concepts to begin the RAG search."
    )


# --- Stage 2: Query Generation ---
class SearchQueries(BaseModel):
    semantic_queries: List[str] = Field(
        description="Natural language questions optimized for vector/similarity search."
    )
    low_level_keywords: List[str] = Field(
        description="Specific exact-match keywords, phrases or boolean combinations that can be directly extracted from the prompt."
    )
    high_level_keywords: List[str] = Field(
        description="Global keywords that relate to the overall topic of the user query."
    )
    reasoning: str = Field(
        description="Brief explanation of why these queries were chosen based on knowledge gaps."
    )


# --- Stage 3: Evaluation ---
class InformationAssessment(BaseModel):
    is_sufficient: bool = Field(
        description="True if the retrieved context contains enough explicit client policy/rules to write a definitive grading prompt. False otherwise."
    )
    reasoning: str = Field(
        description="Explanation of why the information is or is not sufficient."
    )
    identified_gaps: Optional[List[str]] = Field(
        description="If not sufficient, list the exact policy details or definitions still missing."
    )


# --- Stage 4: Final Prompt Output ---
class EnhancedExtractionPrompt(BaseModel):
    parameter_name: str = Field(description="The original parameter name.")
    system_prompt: str = Field(
        description="The role and rule-based system prompt for the extraction LLM."
    )
    evaluation_criteria: List[str] = Field(
        description="Step-by-step criteria derived from the client's documents that the extraction LLM must follow to pass/fail this parameter."
    )
    few_shot_examples: Optional[str] = Field(
        description="Hypothetical examples of pass/fail based on the retrieved context, if applicable."
    )
