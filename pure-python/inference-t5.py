import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Verify GPU availability
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Load the original tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Instructions for use
print("Enter a prompt to receive a generated response. Type 'exit' or 'quit' to end the session.")

# Interactive loop for user prompts
while True:
    # Get user input
    input_text = input("Enter your prompt: ").strip()
    
    # Check for quit sequence
    if input_text.lower() in ['exit', 'quit']:
        print("Exiting the program. Goodbye!")
        break
    
    # Process the input and generate the result
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
    
    # Print the generated response
    print(f"Generated response: {response}")
    print("-" * 50)  # Separator for readability