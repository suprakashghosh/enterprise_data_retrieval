import json
import os

import instructor
from models import *
from prompts import *
from pydantic import BaseModel

client = instructor.from_provider("google/gemini-3-flash-preview",async_client=True)


class PromptEnhancementAgent:
    def __init__(self, max_retries=3):
        # self.llm = llm_client  # Your wrapper for OpenAI/Anthropic etc.
        # self.rag = rag_client  # Your wrapper for Pinecone/Weaviate/FAISS etc.
        self.max_retries = max_retries

    async def get_llm_response_from_instructor(user_input: str,
                                           response_format: BaseModel,
                                           system_prompt:str="You are a helpful assistant",
                                           max_tokens:int=1000):
        response = await client.create(
            messages=[
                {   "role":"system",
                    "content":system_prompt,
                    "role": "user",
                    "content": user_input,
                }
            ],
            response_model=response_format,
            generation_config={
                "temperature": 0,
                "max_tokens": max_tokens,
                "top_p": 1,
                "top_k": 32,
            },
        )
        return response


    async def run(self, basic_parameter_text: str) -> EnhancedExtractionPrompt:
        # State variables
        gathered_context = []
        identified_gaps = []
        tries = 0
        is_sufficient = False

        # --- STAGE 1: Decompose ---
        print(f"Decomposing parameter: {basic_parameter_text}")
        decomposed_param: DecomposedParameter = await self.get_llm_response_from_instructor(
            system_prompt=DECOMPOSER_PROMPT,
            user_input=basic_parameter_text,
            response_format=DecomposedParameter,
        )

        # --- STAGE 2 & 3: Agentic Search & Evaluate Loop ---
        while tries < self.max_retries and not is_sufficient:
            tries += 1
            print(f"--- Search Iteration {tries}/{self.max_retries} ---")

            # Prepare input for query generation
            query_input = {
                "decomposed_parameter": decomposed_param.model_dump(),
                "identified_gaps_from_last_run": identified_gaps,
            }

            # Generate Queries
            queries: SearchQueries = await self.get_llm_response_from_instructor(
                system_prompt=QUERY_GENERATOR_PROMPT,
                user_input=json.dumps(query_input),
                response_format=SearchQueries,
            )

            # Execute RAG (User implemented)
            # Fetch semantic and keyword results, deduplicate, and append to state
            new_context = self.rag.hybrid_search(
                semantic=queries.semantic_queries, keywords=queries.keyword_queries
            )
            gathered_context.extend(new_context)

            # Evaluate Information
            eval_input = {
                "original_parameter": basic_parameter_text,
                "decomposed_intent": decomposed_param.core_objective,
                "gathered_context_so_far": gathered_context,
            }

            assessment: InformationAssessment = await self.get_llm_response_from_instructor(
                system_prompt=EVALUATOR_PROMPT,
                user_input=json.dumps(eval_input),
                response_format=InformationAssessment,
            )

            is_sufficient = assessment.is_sufficient
            identified_gaps = assessment.identified_gaps or []

            print(
                f"Assessment: Sufficient? {is_sufficient}. Reasoning: {assessment.reasoning}"
            )

        # --- STAGE 4: Final Prompt Generation ---
        print("Generating final enhanced prompt...")
        final_input = {
            "parameter": basic_parameter_text,
            "decomposed_details": decomposed_param.model_dump(),
            "validated_client_context": gathered_context,
        }

        # Note: Even if max_retries is hit and it's not "sufficient", we still generate
        # the best possible prompt using whatever context we DID find.
        final_prompt: EnhancedExtractionPrompt = await self.get_llm_response_from_instructor(
            system_prompt=PROMPT_ENGINEER_PROMPT,
            user_input=json.dumps(final_input),
            response_format=EnhancedExtractionPrompt,
        )

        return final_prompt


# Usage:
# agent = PromptEnhancementAgent(llm_client, rag_client)
# result = agent.run("Were the terms and conditions of the product explained to the customer?")
# print(result.system_prompt)
