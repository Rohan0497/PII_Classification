import argparse
from transformers import pipeline
import os
import csv

def setup_pipeline(model_path):
    return pipeline('token-classification', model=model_path, tokenizer=model_path)



def perform_inference(pipeline, text):
    result = pipeline(text)
    # print(result)  
    return result

def save_results(results, output_dir, input_data):
    output_file = os.path.join(output_dir, 'inference_results.csv')
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sample', 'Source Text', 'Predicted Labels'])
        for i, (data, result) in enumerate(zip(input_data, results)):
            # Organize results by entity type, merging tokens appropriately
            organized_results = {}
            for ent in result:
                entity_type = ent['entity'][2:]  # Strip the B- or I-
                word = ent['word'].replace('##', '')
                score = ent['score']
                if score <= 0.5:
                    continue

                # Append word based on entity type characteristics
                if entity_type in ['EMAIL', 'FIRSTNAME', 'LASTNAME', 'USERNAME']:
                    # Append without space if it's part of a continuous entity like email or names
                    if entity_type in organized_results:
                        last_char = organized_results[entity_type]['word'][-1]
                        if last_char.isalnum() and word[0].isalnum():
                            organized_results[entity_type]['word'] += word
                        else:
                            organized_results[entity_type]['word'] += '' + word
                    else:
                        organized_results[entity_type] = {'word': word, 'scores': [score]}
                else:
                    # Normal entities with spaces
                    if entity_type in organized_results:
                        organized_results[entity_type]['word'] += ' ' + word
                    else:
                        organized_results[entity_type] = {'word': word, 'scores': [score]}

                if entity_type not in organized_results:
                    organized_results[entity_type] = {'word': word, 'scores': [score]}
                else:
                    organized_results[entity_type]['scores'].append(score)

            # Format the results for output by averaging scores
            formatted_result = []
            for entity, details in organized_results.items():
                average_score = sum(details['scores']) / len(details['scores'])
                formatted_result.append(f"{entity}: {details['word']} (Score: {average_score:.2f})")

            writer.writerow([i, data, '; '.join(formatted_result)])
    print(f"Results saved to {output_file}")



def main(args):
    os.makedirs(args.outputdir, exist_ok=True)

    nlp_pipeline = setup_pipeline('./output_dir/model/best_model')

    if args.text:        
        results = perform_inference(nlp_pipeline, args.text)
        save_results([results], args.outputdir, [args.text])
    elif args.dataset:
     
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
