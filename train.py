import os
import logging
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from datasets import load_from_disk
from metric import compute_metrics
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# Setup logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure output directories exist
os.makedirs('./output_dir', exist_ok=True)

def main():
    tokenizer = AutoTokenizer.from_pretrained("google/electra-large-generator")
    dataset = load_from_disk('./data/tokenized_dataset')
    
    with open('./data/label2id.json') as f:
        label2id = json.load(f)
    
    model = AutoModelForTokenClassification.from_pretrained(
        "google/electra-large-generator",
        num_labels=len(label2id),
        label2id=label2id,
        id2label={idx: label for label, idx in label2id.items()}
    )
   
    training_args = TrainingArguments(
        output_dir="./output_dir",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./runs",
        report_to='tensorboard'
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

       # Print input sizes for debugging
    # for batch in trainer.get_train_dataloader():
    #     logging.info(f"Input IDs size: {batch['input_ids'].size()}")
    #     if (batch['input_ids'].size(1) > 512):
    #         logging.warning("Found input IDs longer than 512 tokens")


    train_result = trainer.train()
    trainer.save_model("./output_dir/model/best_model")  # Save the best model
    tokenizer.save_pretrained('./output_dir/model')  # Save the tokenizer used with the best model
    evaluation_result = trainer.evaluate()

    with open('./output_dir/model/training_log_history.json', 'w') as log_file:
        json.dump(trainer.state.log_history, log_file)

    # Log and save results
    logging.info(f"Training completed. Metrics: {train_result.metrics}")
    logging.info(f"Evaluation completed. Metrics: {evaluation_result}")





if __name__ == "__main__":
    main()

