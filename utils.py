import pandas as pd
import json


def safe_json_loads(s):
    """Attempt to decode JSON from string 's'. Return None if an error occurs."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

# Load a JSONL file into a DataFrame, handling missing IDs.
def load_jsonl_to_dataframe(filepath):
    with open(filepath, 'r') as file:
        # Read, strip, decode JSON, and filter out None in one go
        data = [safe_json_loads(line.strip()) for line in file if line.strip()]
        data = [item for item in data if item]  # Filter out None values if any remain

    # Convert to DataFrame and set 'id' column, handling missing IDs
    df = pd.DataFrame(data)
    df['id'] = df.get('id', 'No ID found')

    return df

if __name__ == "__main__":
    df = load_jsonl_to_dataframe("statperls_sample_qas.jsonl")
    df.to_csv('train_data/statperls_sample_qas.csv', index=False)