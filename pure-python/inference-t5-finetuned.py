import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Verify GPU availability
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Load fine-tuned tokenizer and model
model_name = "./fine-tuned-model"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Function to load all datasets from a directory
def load_datasets_from_directory(directory):
    prompts, expected_outputs, files = [], [], []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                for line in lines:
                    if '|' in line:
                        input_text, output_text = line.split('|', 1)
                        prompts.append(input_text.strip())
                        expected_outputs.append(output_text.strip())
                        files.append(filename)
                    else:
                        print(f"Skipping malformed line: {line}")
    return prompts, expected_outputs, files

# Directory for validation datasets
val_dir = "validation_datasets"

# Load validation datasets
prompts, expected_outputs, files = load_datasets_from_directory(val_dir)

# Process each prompt and generate the result
print("Starting automated test...")
for i, (input_text, expected_output, file) in enumerate(zip(prompts, expected_outputs, files), 1):
    inputs = tokenizer(input_text, return_tensors='pt', padding='max_length', max_length=50, truncation=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Generate response with attention mask and additional parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  # Pass the attention mask
            max_length=50,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5,  # Use beam search to potentially improve generation quality
            early_stopping=True,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Control randomness
            top_k=50,  # Limit to top-k tokens
            top_p=0.9  # Nucleus sampling
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Check if the response matches the expected output
    if response.strip() != expected_output.strip():
        print(f"WARNING: For prompt from file '{file}': '{input_text}', expected '{expected_output}' but got '{response}'")
    print(f"Test iteration {i} completed for file '{file}'.")

print("Automated test completed.")
print("Enter your own prompts. To exit, type 'exit' or 'quit'.")

while True:
    user_input = input("Enter a new prompt: ").strip()
    
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the program. Goodbye!")
        break

    inputs = tokenizer(user_input, return_tensors='pt', padding='max_length', max_length=50, truncation=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  # Pass the attention mask
            max_length=50,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=5,  # Use beam search to potentially improve generation quality
            early_stopping=True,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Control randomness
            top_k=50,  # Limit to top-k tokens
            top_p=0.9  # Nucleus sampling
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Input text: {user_input}")
    print(f"Generated response: {response}")
    print("-" * 50)  # Separator for readability
