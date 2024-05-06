import argparse
from transformers import pipeline
import os
import csv

def setup_pipeline(model_path):
    # Set up the pipeline for token classification using the specified model
    return pipeline('token-classification', model=model_path, tokenizer=model_path)

# def perform_inference(pipeline, text):
#     # Use the pipeline to perform inference on input text or list of texts
#     if isinstance(text, list):
#         return [pipeline(t) for t in text]
#     else:
#         return pipeline(text)


def perform_inference(pipeline, text):
    # Use the pipeline to perform inference on input text or list of texts
    result = pipeline(text)
    print(result)  # Add this line to inspect the output structure
    return result

def save_results(results, output_dir, input_data):
    output_file = os.path.join(output_dir, 'inference_results.csv')
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sample', 'Source Text', 'Predicted Labels'])
        for i, (data, result) in enumerate(zip(input_data, results)):
            # Filter results to include only high-confidence predictions
            filtered_result = [ent for ent in result if ent['score'] > 0.5]
            # Format the results for output
            formatted_result = '; '.join([f"{ent['entity']}: {ent['word']} (Score: {ent['score']:.2f})" for ent in filtered_result])
            writer.writerow([i, data, formatted_result])
    print(f"Results saved to {output_file}")


def main(args):
    os.makedirs(args.outputdir, exist_ok=True)

    nlp_pipeline = setup_pipeline('./output_dir/model/best_model')

    if args.text:
        # Direct text input from the user
        results = perform_inference(nlp_pipeline, args.text)
        save_results([results], args.outputdir, [args.text])
    elif args.dataset:
        # Load data from a provided file path (assuming one text per line)
        with open(args.dataset, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
        results = perform_inference(nlp_pipeline, lines)
        save_results(results, args.outputdir, lines)
    else:
        print("No input provided. Please specify text or a dataset path.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform inference with a trained model on a provided text or dataset.')
    parser.add_argument('--outputdir', type=str, required=True, help='Directory to save the inference outputs.')
    parser.add_argument('--text', type=str, help='Direct text input for inference.')
    parser.add_argument('--dataset', type=str, help='Path to a text file containing data for inference.')
    args = parser.parse_args()

    main(args)
