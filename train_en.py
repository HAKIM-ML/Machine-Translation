import pandas as pd
from datasets import Dataset
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration, TrainingArguments, Trainer
import torch

class TranslationModelTrainer:
    def __init__(self, dataset_path, model_name, tokenizer_name):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_data()
        self.load_tokenizer()
        self.load_model()

    def load_data(self):
        df = pd.read_csv(self.dataset_path)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df.head(2000)
        self.dataset = Dataset.from_pandas(df)

    def load_tokenizer(self):
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.tokenizer_name)

    def load_model(self):
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        if torch.cuda.is_available():
            print("Model moved to CUDA device.")
        else:
            print("CUDA is not available. Moving the model to CPU.")

    def preprocess_function(self, examples):
        source_lang = "en"
        target_lang = "bn"
        self.tokenizer.src_lang = source_lang

        inputs = self.tokenizer(examples['en'], max_length=128, truncation=True, padding="max_length")
        targets = self.tokenizer(examples['bn'], max_length=128, truncation=True, padding="max_length", return_tensors="pt")

        inputs["labels"] = targets["input_ids"]
        inputs["labels"] = [
            [(label if label != self.tokenizer.pad_token_id else -100) for label in labels_example]
            for labels_example in inputs["labels"]
        ]

        return inputs

    def tokenize_dataset(self):
        self.tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True, remove_columns=self.dataset.column_names)

    def train_model(self):
        training_args = TrainingArguments(
            output_dir= 'Machine Translation',
            evaluation_strategy='epoch',
            learning_rate=3e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            weight_decay=0.01,
            save_total_limit=3,
            save_steps=500,
            logging_steps=100,
            fp16=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets.shuffle(seed=42).select(range(1500)),
            eval_dataset=self.tokenized_datasets.shuffle(seed=42).select(range(1500, 2000)),
        )

        trainer.train()

    def save_model(self, model_path, tokenizer_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)

    def run(self, model_path="finetuned_model_en", tokenizer_path="finetuned_tokenizer_en"):
        self.tokenize_dataset()
        self.train_model()
        self.save_model(model_path, tokenizer_path)

# 
trainer = TranslationModelTrainer(dataset_path='final_data.csv', model_name='model', tokenizer_name='tokenizer')
trainer.run()
