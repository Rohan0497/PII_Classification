# PII Detection Model

## Overview

This repository contains scripts designed to train, evaluate, and infer a machine learning model capable of detecting Personally Identifiable Information (PII) in textual data. 



## Installation

To set up the project environment, follow these steps:


1. **Install Required Libraries**:

    ```bash
    pip install -r requirements.txt
    ```

## Structure

- `data_prep.py`: Prepares and processes the data for training.
- `train.py`: Contains the training loop and saves the model.
- `metric.py`: Evaluates the model's performance using various metrics.
- `inference.py`: Performs inference on new data to detect PII.
- `post_processing.py`: Processes the output from the inference to clean and format the results.

## Usage

2. **Data Preparation**:

    ```bash
    python data_prep.py
    ```

3.  **Training**:

    ```bash
    python train.py
    ```


4. **Evaluating the Model**:

    ```bash
    python metric.py
    ```

Before running inference.py file make sure you run the data_prep.py file if you intend to use the best model directly.

5.  **Running Inference**:

    ```bash
    python inference.py --modelpath /path to best model --outputdir ./path_to_output_directory
    ```
    
    ```bash
    modelpath = ./ouput_dir/model/best_model  . Use this path if you are using model from google drive link 
    ```
    
After you run the inference.py file, it will save an inference_results.csv in the output directory you specified. 

For simplicity, let's assume that if you specify the output directory as ./output, then the file will be saved at the location ./output/inference_results.csv.

For post-processing of the file, make sure to specify the location of the inference_results.csv file as --input_file_location, and then specify the output location. 

Demo implementation would be as follows

post_processing.py --input_file ./output/inference_results.csv --outputdir ./output

6.  **Post Processing**:

    ```bash
    python post_processing.py --input_file ./path_to_input_file/inference_results.csv --outputdir ./path_to_output_directory
    ```



7. **Custom Text Inference**



    ```bash
    python custom_inference.py --modelpath /path to best model --outputdir /path to output_directory --text "Your input text here"
    ```

    ```bash
    modelpath = ./ouput_dir/model/best_model  . Use this path if you are using model from google drive link 
    ```
   
   To run the inference script on a custom piece of text, use the following command structure. Replace /path/to/output_directory with the actual directory path where you want the inference results to be saved, 
    and replace Your input text here with the text you want to analyze.

### Assumptions
The model deals  with text data.



### Dataset Truncation
The dataset was sourced from Hugging Face and originally comprised 54 Personal Identifiable Information (PII) classes with approximately 59,000 sample texts. To expedite experimentation and reduce computational requirements, the dataset was truncated to include only 10,000 sample texts. 

### Output Format
The final output of the process is a CSV file containing cleaned and structured data extracted from textual sources. Below is a detailed explanation of the columns present in the output CSV file:

Sample: An index or identifier for each entry in the dataset.

Source Text: The original text from which information has been extracted.

Values: A semicolon-separated list of key-value pairs where each key represents a type of information identified in the text, and the value is the corresponding extracted data.




