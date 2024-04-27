from selfcheckgpt.modeling_selfcheck import SelfCheckNLI, SelfCheckLLMPrompt
from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
import torch
import os
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from .api_config import HuggingFaceConfig, OpenAIConfig

class SelfCheckGPT:
    """
    A class to evaluate the quality of generated samples using the SelfCheckGPT.
    A higher score indicates higher chance of being hallucination.
    SelfCheckGPT repository: https://github.com/potsawee/selfcheckgpt
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ['HF_TOKEN'] = HuggingFaceConfig.HF_TOKEN
        os.environ['OPENAI_API_KEY'] = OpenAIConfig.OpenAI_API_KEY
    
    def evaluate(self,  generations: str, samples: list, option = "LLM"):
        """
        Evaluate the model on the generated samples.
        """
        sentences = sent_tokenize(generations)
        samples = samples

        # Initialize the model
        if option == "NLI":
            selfcheck = SelfCheckNLI(device=self.device)
            scores = selfcheck.predict(
            sentences = sentences,                       
            sampled_passages = samples
            )

        elif option == "LLM":
            selfcheck = SelfCheckAPIPrompt(client_type="openai", model="gpt-3.5-turbo")
            # llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
            # selfcheck = SelfCheckLLMPrompt(llm_model, self.device)
            scores = selfcheck.predict(
            sentences = sentences,                       
            sampled_passages = samples,
            verbose=True
            )
        
        else:
            raise ValueError("Invalid option. Please choose between 'NLI' and 'LLM'.")
    
        
        avg_score = sum(scores) / len(scores)
        
        return scores, avg_score
    

if __name__ == "__main__":
    generation = "Non-surgical management strategies for Morton neuroma include changes in shoe wear, weight management, physical therapy, and the use of warm compresses and ice."
    sample1 =  "Non-invasive treatment options for Morton neuroma consist of modifying footwear, managing weight, engaging in physical therapy, and applying warm and cold compresses."
    sample2 = "Non-surgical treatment options for Morton neuroma involve adjusting footwear, managing body weight, engaging in physical therapy, and applying warm compresses and ice."
    sample3 = "Non-surgical approaches to treating Morton's neuroma involve modifying footwear, managing weight, engaging in physical therapy, and applying warm compresses and ice."
    samples = [sample1, sample2, sample3]

    # Initialize the SelfCheckGPT model
    selfcheckgpt = SelfCheckGPT()
    
    # Evaluate the model on the generated samples
    scores_LLM, avg_score_LLM,  = selfcheckgpt.evaluate(generation, samples, option="LLM")
    # scores_ntl, avg_score_ntl = selfcheckgpt.evaluate(generation, samples, option="NLI")
    
    print("Scores for LLM model:", scores_LLM)
    print("Average score for LLM model:", avg_score_LLM)
    # print("Scores for NLI model:", scores_ntl)
    # print("Average score for NLI model:", avg_score_ntl)