import pandas as pd
import logging
from transformers import AutoTokenizer
from datasets import Dataset
import itertools
import collections
from tqdm.auto import tqdm
import json
import os

# Set up logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

os.makedirs('./data', exist_ok=True)

def load_data(filepath):
    try:
        df = pd.read_json(filepath, lines=True)
        df = df.rename(columns={
            "masked_text": "target_text",
            "unmasked_text": "source_text",
            "tokenised_unmasked_text": "tokenized_text",
            "token_entity_labels": "ner_tags"
        }).dropna().reset_index(drop=True)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise



def filter_token_lengths(df, tokenizer):
    try:
        # Handling pre-tokenized data 
        tt_lens = [len(tokenizer(t, is_split_into_words=isinstance(t, list), truncation=True, max_length=512)['input_ids']) for t in tqdm(df['tokenized_text'].tolist())]
        df['tt_lens'] = tt_lens
        df = df[df['tt_lens'] <= 510].reset_index(drop=True)
        logging.info("Filtered token lengths to a maximum of 510.")
        return df
    except Exception as e:
        logging.error(f"Failed to filter token lengths: {e}")
        raise

def prepare_labels(df):
    try:
        all_tags = list(itertools.chain.from_iterable(df['ner_tags'].tolist()))
        tag_counts = collections.Counter(all_tags)
        all_labels = list(tag_counts.keys())
        label2id = {label: idx for idx, label in enumerate(all_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        df['ner_tags'] = [
            [label2id[tag] for tag in tags]
            for tags in df['ner_tags']
        ]
        logging.info("Label dictionaries prepared.")
        return label2id, id2label
    except Exception as e:
        logging.error(f"Failed to prepare labels: {e}")
        raise


def validate_lengths(df):
    try:
        
        df['source_words'] = [text.split() for text in df['source_text']]
        # Check if the length of the list in 'tokenized_text' matches the length of 'ner_tags'
        idx = [i for i in range(len(df)) if len(df.loc[i, 'tokenized_text']) != len(df.loc[i, 'ner_tags'])]
        df.drop(index=idx, inplace=True)
        df.reset_index(drop=True, inplace=True)
        logging.info(f"Validated lengths and removed {len(idx)} discrepancies.")
    except Exception as e:
        logging.error(f"Error validating lengths of tokenized texts and tags: {e}")
        raise

def visualize_data_structures(label2id, dataset):
    # Print label mappings
    print("Label to ID Mapping:", label2id)

    # Convert a small portion of the dataset to a pandas DataFrame for visualization
    # sample_df = dataset.to_pandas().head()  # Convert the first few entries to a DataFrame
    print("Dataset structure:")
    # print(sample_df)
    print(dataset)
   

def truncate_data(df):
    try:
        # Truncate the dataframe to only use the first 1000 samples
        df = df.head(10000)
        logging.info(f"Data truncated to the first 1000 samples.")
        return df
    except Exception as e:
        logging.error(f"Failed to truncate data: {e}")
        raise    

def align_labels(example, tokenizer):
    
    tokenized_input = tokenizer(example["tokenized_text"], is_split_into_words=True, truncation=True, max_length=512, padding="max_length",return_attention_mask=True)
    word_ids = tokenized_input.word_ids()
    aligned_labels = [-100 if i is None else example["ner_tags"][i] for i in word_ids]
    example['labels'] = aligned_labels
    example['input_ids'] = tokenized_input['input_ids']
    example['attention_mask'] = tokenized_input['attention_mask']
  
    return example


def main():
    filepath = "./data/pii200k_english.jsonl"
    tokenizer = AutoTokenizer.from_pretrained("google/electra-large-generator")
    
    df = load_data(filepath)
   
    df = truncate_data(df)
    df = filter_token_lengths(df, tokenizer)
    
    label2id, id2label = prepare_labels(df)
  
    validate_lengths(df)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda example: align_labels(example, tokenizer), num_proc=8,remove_columns=['tokenized_text', 'ner_tags', 'tt_lens'])
    tokenized_dataset = dataset.train_test_split(test_size=0.2)    
    
    # visualize_data_structures(label2id,tokenized_dataset)
    tokenized_dataset.save_to_disk("./data/tokenized_dataset")  # Save train and test dataset
    json.dump(label2id, open('./data/label2id.json', 'w'))
    json.dump(id2label, open('./data/id2label.json', 'w'))

    logging.info("Preprocessing complete and data saved. Ready for model training.")

  

if __name__ == "__main__":
    main()
