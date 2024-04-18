from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import torch

if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        raise EnvironmentError("Apple MPS not available.")

    model = AutoModelForCausalLM.from_pretrained(
        "Orangejustin/llama-2-7b-qa-generator",
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/llama-2-7b-chat-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Run text generation pipeline with our next model
    instruction = ("You are a synthetic question-answer pair generator. Given a chunk of context about some topic(s), "
                       "generate 1 example question a user could ask and would be answered using information from the "
                       "chunk: ")
    chunk = "A wiki (/ˈwɪki/ ⓘ WI-kee) is a form of online hypertext publication that is collaboratively edited and managed by its own audience directly through a web browser. A typical wiki contains multiple pages for the subjects or scope of the project, and could be either open to the public or limited to use within an organization for maintaining its internal knowledge base."

    prompt = f"{instruction}{chunk}"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=300, device=torch.device("mps"))
    print("Inference starts...")
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])
