import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from datasets import load_from_disk
import os
import csv

def load_model(model_path):
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def perform_inference(model, tokenizer, dataset):

    results = []
    for i, example in enumerate(dataset):
        # Directly use the pre-tokenized and pre-processed data
        inputs = {
            'input_ids': torch.tensor(example['input_ids']).unsqueeze(0),             
            'attention_mask': torch.tensor(example['attention_mask']).unsqueeze(0)
        }
        with torch.no_grad():
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])  # Convert input_ids to tokens
        predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]
        results.append((i, example['source_text'], tokens, predicted_labels))
    return results


def save_results(results, output_dir):
    output_file = os.path.join(output_dir, 'inference_results.csv')
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sample', 'Source Text', 'Tokens', 'Predicted Labels'])
        for sample_id, source_text, tokens, labels in results:
            writer.writerow([sample_id, source_text, ' '.join(tokens), ' '.join(labels)])
    print(f"Results saved to {output_file}")


def main(args):
    
    os.makedirs(args.outputdir, exist_ok=True)

    # model, tokenizer = load_model('./output_dir/model/best_model')
    model, tokenizer = load_model(args.modelpath)
    # dataset = load_from_disk('./data/tokenized_dataset')
    dataset = load_from_disk(args.dataset)
    results = perform_inference(model, tokenizer, dataset['test'])
    
    save_results(results, args.outputdir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform inference with a trained model on a dataset.')
    parser.add_argument('--modelpath', type=str, required=True, help='Path to the trained model directory.')
    parser.add_argument('--datasetlocation', type=str, required=True, help='Path to the processed dataset directory.')
    parser.add_argument('--outputdir', type=str, required=True, help='Directory to save the inference outputs.')
    args = parser.parse_args()

    main(args)
