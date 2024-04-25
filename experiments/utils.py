import json
def read_jsonl(file_path):
    """Read a JSONL file and return the data."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            content = json.loads(line)
            data.append(content)  
    return data