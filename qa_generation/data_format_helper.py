import json

def reset_ids(input_file_path, output_file_path):
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for i, line in enumerate(infile):
            json_obj = json.loads(line)
            json_obj['id'] = i  # Update the ID
            json_str = json.dumps(json_obj)
            outfile.write(json_str + '\n')

    print("Finished resetting IDs.")


def remove_duplicates(input_file_path, output_file_path):
    seen_questions = set()
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            json_obj = json.loads(line)
            question = json_obj.get('question')  
            if question not in seen_questions: 
                seen_questions.add(question)  
                json_str = json.dumps(json_obj)
                outfile.write(json_str + '\n') 

    print("Finished removing duplicates.")
    