from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import json
import time
from openai_config import OpenAIConfig
import os
from tqdm import tqdm
import pandas as pd
from threading import Lock

def _extract_question_and_answer(response_content: str) -> tuple[str, str]:
    """
    Extract the question and answer from the response content.
    """
    question = response_content.split("<QUESTION>: ")[1].split("<ANSWER>: ")[0].strip()
    answer = response_content.split("<ANSWER>: ")[1].strip()

    return question, answer

class HallucinationQAPairsGenerator:
    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model
        self.client = None

    def init_openai_client(self):
        """
        Initialize the OpenAI client.
        """
        # Load the OpenAI API key
        api_key = OpenAIConfig.PROXY_API_KEY
        # base_url = OpenAIConfig.PROXY_BASE_URL

        # Initialize the OpenAI client
        self.client = OpenAI(api_key=api_key)

    def generate_qa_pair(self) -> tuple[str, str]:
        """
        Generate a question-answer pair.
        """
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": "You are a hallucination synthetic question-answer pair generator. Your objective is to create a question related to medical science that is unanswerable and to provide a hallucinated answer that sounds plausible but is factually incorrect. Generate the question-answer pair in the following format: <QUESTION>: $question <ANSWER>: $answer. The answer should be succinct and sound plausible, yet be factually incorrect."},
        #         {"role": "user", "content": "Generate a question-answer pair related to medical science that fits the specified criteria."}
        #         ],
        #         n=1,
        #         temperature=0.8
        #     )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a hallucination synthetic medical question-answer pair generator. Your objective is to create a question that is unanswerable and to provide a hallucinated answer that sounds plausible but is factually incorrect. Generate the question-answer pair in the following format: <QUESTION>: $question <ANSWER>: $answer. The answer should be succinct and sound plausible, yet be factually incorrect."},
                {"role": "user", "content": "Generate a question-answer pair related to medical science that fits the specified criteria."}
                ],
                n=1,
                temperature=0.8
            )

        response_content = response.choices[0].message.content

        question, answer = _extract_question_and_answer(response_content)

        return question, answer
    

    def generate_qa_pairs(self, output_file:str, num_qa: int) -> None:
        """
        Generate multiple question-answer pairs.
        """

        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(num_qa):
                question, answer = self.generate_qa_pair()
                # print("=====================================")
                # print("Question: ", question)
                # print("Answer: ", answer)
                # print("=====================================")
                data = {
                    "id" : i,
                    "question" : question,
                    "answer" : answer,
                    "covered" : False
                }

                f.write(json.dumps(data) + '\n')


if __name__ == "__main__":
    hqag = HallucinationQAPairsGenerator(model="gpt-4-turbo")
    hqag.init_openai_client()
    
    # Output directory
    output_dir = "PAQ/hallucination_qa/fully_hallucination_qa.jsonl"

    # Generate question-answer pairs
    hqag.generate_qa_pairs(output_file=output_dir, num_qa=1500)