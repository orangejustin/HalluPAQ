import importlib.resources
from pathlib import Path
import pathlib
# Path to the data file within the package
# data_path = importlib.resources.files("FactScoreLite") / "data"
data_path = pathlib.Path(__file__).parent / "data"

# Path to the data file within the package
atomic_facts_demons_path = data_path / "atomic_facts_demons.json"
fact_scorer_demons_path = data_path / "fact_scorer_demons.json"

# OpenAI API
max_tokens = 1024
temp = 0.7
model_name = "gpt-3.5-turbo"

# Database path
# Current folder
# current_folder = Path(__file__).parent
# facts_db_path = current_folder / "database/facts.json"
# decisions_db_path = current_folder / "database/decisions.json"
