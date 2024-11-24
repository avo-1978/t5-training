import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_callback import TrainerCallback

# Verify GPU availability
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create training and validation directories if they don't exist
train_dir = "training_datasets"
val_dir = "validation_datasets"
create_directory(train_dir)
create_directory(val_dir)

# Function to load all datasets from a directory
def load_datasets_from_directory(directory, tokenizer):
    inputs, labels = [], []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                for line in lines:
                    if '|' in line:
                        input_text, output_text = line.split('|', 1)
                        input_ids = tokenizer.encode(input_text.strip(), return_tensors='pt').squeeze()
                        output_ids = tokenizer.encode(output_text.strip(), return_tensors='pt').squeeze()
                        inputs.append(input_ids)
                        labels.append(output_ids)
                    else:
                        print(f"Skipping malformed line: {line}")
    return inputs, labels

# Define the model directory and model name
model_dir = "./models/t5-small"

# Check if the model directory exists
if not os.path.exists(model_dir):
    print(f"Model directory '{model_dir}' not found. Please ensure the models are downloaded and placed in the 'models' subdirectory.")
    print("You can download the models from Hugging Face and place them in the 'models/t5-small' directory.")
    exit(1)

# Load the tokenizer and model from the local directory
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Load training and validation datasets
train_inputs, train_labels = load_datasets_from_directory(train_dir, tokenizer)
val_inputs, val_labels = load_datasets_from_directory(val_dir, tokenizer)

# Convert to dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {'input_ids': self.inputs[idx], 'labels': self.labels[idx]}

train_dataset = CustomDataset(train_inputs, train_labels)
val_dataset = CustomDataset(val_inputs, val_labels)

# Data collator with padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Early stopping callback based on validation loss
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, threshold):
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and metrics.get("eval_loss") is not None:
            eval_loss = metrics["eval_loss"]
            if eval_loss < self.threshold:
                print(f"Validation loss has reached the threshold ({eval_loss} < {self.threshold}). Stopping training.")
                control.should_training_stop = True

# Fine-tuning configuration
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=3,
    save_steps=100,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=25,
)

# Create Trainer instance with early stopping callback
early_stopping_callback = EarlyStoppingCallback(threshold=0.001)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[early_stopping_callback],  # Add the early stopping callback
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")
