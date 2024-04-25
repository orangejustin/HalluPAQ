from FactScoreLite.factscore import FactScore
import os
from api_config import OpenAIConfig

class FactScoreEvaluator:
    """
    A class to evaluate the quality of generated samples using the FactScore.
    A higher score indicates lower chance of being hallucination.
    FactScore original repository: https://github.com/shmsw25/FActScore/tree/main
    FactScoreLite repository: https://github.com/armingh2000/FactScoreLite/tree/main?tab=readme-ov-file
    """
    def __init__(self, gamma:int = 10, db_id:str = ""):
        os.environ['OPENAI_API_KEY'] = OpenAIConfig.OpenAI_API_KEY
        self.fact_score = FactScore(gamma, db_id)

    def get_batch_fact_score(self, generations:list[str], knowledge_sources:list[str]) -> tuple[list[float], list[float]]:
        scores, init_scores = self.fact_score.get_factscore(generations, knowledge_sources)
        self.fact_score.delete_db()

        return scores, init_scores
    
    def get_single_fact_score(self, generation:str, knowledge_source:str) -> tuple[float, float]:
        score, init_score = self.fact_score.get_factscore([generation], [knowledge_source])
        self.fact_score.delete_db()
        return score, init_score

if __name__ == '__main__':
    generation1 = "Non-surgical management strategies for Morton neuroma include changes in shoe wear, weight management, physical therapy, and the use of warm compresses and ice."
    generation2 = "The spleen plays a critical role in regulating core body temperature by releasing a hormone called thermulin, which adjusts blood flow and metabolic rate to maintain temperature balance within the body."
    knowledge_sources = "Morton Neuroma -- Enhancing Healthcare Team Outcomes. Morton neuroma is best managed non-surgically with an interprofessional team of healthcare professionals, including a podiatrist, orthopedic surgeon, sports physician, nurse practitioner, and primary care provider. The patient may require pain medications, but the key is changes in shoe wear. The nurse should encourage the patient to wear appropriate, well-padded, non-constrictive shoewear. Obese patients may benefit from weight loss, so a dietary consult is appropriate. The patient may benefit from physical therapy and the use of warm compresses and ice to ease the pain."
    
    # Get the fact score for the covered data
    factscore = FactScoreEvaluator(db_id="demo1")
    score1, init_score1 = factscore.get_single_fact_score(generation1, knowledge_sources)
    

    # Get the fact score for the uncovered data
    factscore = FactScoreEvaluator(db_id="demo2")
    score2, init_score2 = factscore.get_single_fact_score(generation2, knowledge_sources)

    print("Fact score for covered data:", score1, init_score1)
    print("Fact score for uncovered data:", score2, init_score2)
