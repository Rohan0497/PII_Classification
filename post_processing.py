import csv
import argparse
import pandas as pd
import os

def extract_values(tokens, labels):
    """Extract and format values from token-label pairs, ensuring all tokens are cleaned."""
    values = {}
    current_label = None
    current_value = []

    for token, label in zip(tokens, labels):
        # Clean the token: remove '##' and special tokens like '[PAD]', '[CLS]', '[SEP]'
        token = token.replace('##', '')
        if token in ['[PAD]', '[CLS]', '[SEP]']:
            continue

        # Process labels and tokens
        if label.startswith('B-') or (label.startswith('I-') and label[2:] != current_label):
            if current_value and current_label:
                # Join the current tokens into a single string and store under the current label
                values[current_label] = values.get(current_label, []) + [''.join(current_value)]
            current_label = label[2:]  # Start new label
            current_value = [token]
        elif label.startswith('I-') and label[2:] == current_label:
            current_value.append(token)
        elif label == 'O' and current_label:
            if current_value:
                # Store the concatenated sequence for the last label
                values[current_label] = values.get(current_label, []) + [''.join(current_value)]
            current_label, current_value = None, []

    # Store the final sequence if any
    if current_label and current_value:
        values[current_label] = values.get(current_label, []) + [''.join(current_value)]

    # Format the values for output
    return '; '.join(f"{key}: {' | '.join(vals)}" for key, vals in values.items())

def postprocess(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        headers = next(reader)
        writer.writerow(headers + ['Values'])

        for row in reader:
            sample_id, source_text, tokens, labels = row
            tokens = tokens.split()
            labels = labels.split()
            values = extract_values(tokens, labels)
            writer.writerow([sample_id, source_text, ' '.join(tokens), ' '.join(labels), values])

def clean_and_save_dataframe(input_file, output_dir):
    output_file = os.path.join(output_dir, 'cleaned_inference_results.csv')
    df = pd.read_csv(input_file)
    df['Values'] = df['Values'].apply(lambda x: x.replace('##', '') if isinstance(x, str) else x)
    if 'Predicted Labels' in df.columns:
        df = df.drop(columns=['Predicted Labels'])
    if 'Tokens' in df.columns:
        df = df.drop(columns=['Tokens'])
    df.to_csv(output_file, index=False)

def main(args):
    cache_directory = './cache'
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)
    
    intermediate_file = os.path.join(cache_directory, 'processed_inference_results.csv')
    
    # Processing the results
    postprocess(args.input_file, intermediate_file)

    # Cleaning and saving the results
    clean_and_save_dataframe(intermediate_file, args.outputdir)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process inference results, clean them, and save the final output.')
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV file for processing.')
    parser.add_argument('--outputdir', type=str, required=True, help='Final output CSV file after cleaning and processing.')
    
    args = parser.parse_args()
    main(args)




