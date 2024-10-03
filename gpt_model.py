import streamlit as st
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel
import torch
import pandas as pd
from datasets import Dataset, load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import re
from textblob import TextBlob
import time

# Define global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Function to encode text into vectors using Bert
def encode_text(texts, tokenizer, model, batch_size=4):
    if isinstance(texts, str):
        texts = [texts]  # Convert single string to list of one string
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    return embeddings

# Function to create FAISS index
def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

# Function to save dataset and FAISS index
def save_dataset(data, instruction_embeddings, response_embeddings, dataset_path, index_file='my_index.faiss'):
    titles = ["Instruction"] * len(data['Instruction']) + ["Response"] * len(data['Response'])
    texts = data['Instruction'].tolist() + data['Response'].tolist()
    embeddings = np.vstack([instruction_embeddings, response_embeddings])

    # Create a Dataset from the DataFrame
    dataset = Dataset.from_dict({"title": titles, "text": texts, "embeddings": embeddings.tolist()})
    
    # Save the dataset to disk
    dataset.save_to_disk(dataset_path)
    
    # Create and save FAISS index
    index = create_faiss_index(embeddings)
    faiss.write_index(index, index_file)

# Function to load dataset and FAISS index
def load_dataset(dataset_path, index_file='my_index.faiss'):
    # Load the dataset from disk
    dataset = load_from_disk(dataset_path)
    
    # Load the FAISS index for the embeddings
    faiss_index = faiss.read_index(index_file)
    return dataset, faiss_index

# Load model and tokenizer based on selection
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return tokenizer, model

def remove_redundancies(text):
    sentences = text.split('. ')
    unique_sentences = list(dict.fromkeys(sentences))
    return '. '.join(unique_sentences)

def correct_grammar(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

def capitalize_sentences(text):
    if not isinstance(text, str):
        return text
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s.capitalize() for s in sentences]
    capitalized_text = ' '.join(sentences)
    return capitalized_text

def clean_text_spacing(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\. ', '.\n', text)
    return text

def fix_punctuation(text):
    text = re.sub(r'\.\s+', '. ', text)
    text = re.sub(r'\s+([?.!,"])', r'\1', text)
    text = re.sub(r'([?.!,"])', r' \1', text)
    return text

def remove_repetitions(text):
    sentences = text.split('. ')
    seen = set()
    unique_sentences = []
    
    for sentence in sentences:
        stripped_sentence = sentence.strip()
        if stripped_sentence not in seen:
            seen.add(stripped_sentence)
            unique_sentences.append(stripped_sentence)
    
    return '. '.join(unique_sentences) + '.' if unique_sentences else ''

def post_process_gemma_response(text):
    text = capitalize_sentences(text)
    text = correct_grammar(text)
    text = remove_redundancies(text)
    text = clean_text_spacing(text)
    text = fix_punctuation(text)
    text = remove_repetitions(text)
    return text

def generate_response(tokenizer, model, text, model_name):
    start_time = time.time()  # Start timing
    if not isinstance(text, str):
        text = str(text)  # Ensure input is a string
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
    
    # Adjust settings based on model name
    if model_name == "gpt2":
        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,  # Adjusted token limit
            num_beams=3,         # Adjusted beam search
            no_repeat_ngram_size=2, 
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    elif model_name == "google/gemma-2b-it":

        outputs = model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],  # Pass the attention mask here
            max_new_tokens=100,   # Adjusted token limit
            num_beams=3,        # Greedy decoding
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    elif model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        outputs = model.generate(
            input_ids= inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],  # Pass the attention mask here
            max_new_tokens=100,  # Adjusted token limit
            num_beams=3,        # Greedy decoding
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model.generate(inputs['input_ids'],
            max_new_tokens=1000,        # Number of tokens to generate
            no_repeat_ngram_size=2,     # Prevent repeating n-grams of size 2
            do_sample=True,             # Sample instead of using greedy decoding
            top_k=50,                   # Use top-k sampling
            top_p=0.95,                 # Use top-p (nucleus) sampling
            temperature=0.7)
    else:
        outputs = model.generate(inputs['input_ids'], max_new_tokens=50)  # General setting with reduced tokens

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time = time.time()  # End timing
    time_taken = end_time - start_time  # Calculate time taken
    return generated_text, time_taken

# Function to compute ROUGE scores
def compute_rouge_scores_for_each_generated_response(reference_text, generated_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    return scores
def compute_average_rouge_scores_against_test_set(rouge_scores_list):
    if not rouge_scores_list:
        return {"rouge1": {"precision": 0, "recall": 0, "fmeasure": 0},
                "rouge2": {"precision": 0, "recall": 0, "fmeasure": 0},
                "rougeL": {"precision": 0, "recall": 0, "fmeasure": 0}}
    
    # Initialize sums
    sums = {key: {"precision": 0, "recall": 0, "fmeasure": 0} for key in rouge_scores_list[0].keys()}
    
    # Calculate sums
    for score in rouge_scores_list:
        for key in score.keys():
            sums[key]["precision"] += score[key].precision
            sums[key]["recall"] += score[key].recall
            sums[key]["fmeasure"] += score[key].fmeasure
    
    # Calculate averages
    num_scores = len(rouge_scores_list)
    avg_scores = {key: {
        "precision": sums[key]["precision"] / num_scores,
        "recall": sums[key]["recall"] / num_scores,
        "fmeasure": sums[key]["fmeasure"] / num_scores
    } for key in sums}
    
    return avg_scores

def format_rouge_scores(rouge_scores_list):
    if not rouge_scores_list:
        return "No ROUGE scores to display."
    # Initialize sums
    sums = {key: {"precision": 0, "recall": 0, "fmeasure": 0} for key in rouge_scores_list[0].keys()}
    
    # Calculate sums
    for score in rouge_scores_list:
        for key in score.keys():
            sums[key]["precision"] += score[key].precision
            sums[key]["recall"] += score[key].recall
            sums[key]["fmeasure"] += score[key].fmeasure
    
    # Calculate averages
    num_scores = len(rouge_scores_list)
    avg_scores = {key: {
        "precision": sums[key]["precision"] / num_scores,
        "recall": sums[key]["recall"] / num_scores,
        "fmeasure": sums[key]["fmeasure"] / num_scores
    } for key in sums}
    
    # Format output
    formatted_scores = []
    for key, metrics in avg_scores.items():
        formatted_scores.append(f"{key}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['fmeasure']:.4f}")
    
    return "\n".join(formatted_scores)

def main():
    st.title("DBS Connect: Navigating Your Path to Dublin Business School")
    
    # Upload training and testing CSV files
    train_file = st.file_uploader("Upload a Training CSV file", type="csv")
    test_file = st.file_uploader("Upload a Testing CSV file", type="csv")
    
    # Model selection
    model_name = st.selectbox("Select a model", ["gpt2", "google/gemma-2b-it", "meta-llama/Meta-Llama-3-8B-Instruct","mistralai/Mistral-7B-Instruct-v0.3"])
    
    # Initialize performance tracking
    performance_data = []
    
    # Load model button
    if st.button("Load Model"):
        with st.spinner(f"Loading {model_name} model..."):
            gpt_tokenizer, gpt_model = load_model(model_name)
        st.success(f"Model {model_name} loaded successfully!")
        
        if train_file is not None and test_file is not None:
            # Load the training data
            train_data = pd.read_csv(train_file)
            texts = train_data['Instruction'].tolist() + train_data['Response'].tolist()
            
            # Measure embedding generation time
            st.write("Generating embeddings for training data...")
            start_time = time.time()
            embeddings = encode_text(texts, bert_tokenizer, bert_model)
            embedding_time = time.time() - start_time
            
            # Measure FAISS index creation time
            start_time = time.time()
            instruction_embeddings = encode_text(train_data['Instruction'].tolist(), bert_tokenizer, bert_model)
            response_embeddings = encode_text(train_data['Response'].tolist(), bert_tokenizer, bert_model)
            dataset_path = 'my_dataset'
            index_file = 'my_index.faiss'
            save_dataset(train_data, instruction_embeddings, response_embeddings, dataset_path, index_file)
            faiss_creation_time = time.time() - start_time
            
            # Record performance data
            performance_data.append({
                'Model': model_name,
                'Embedding Generation Time (s)': embedding_time,
                'FAISS Index Creation Time (s)': faiss_creation_time
            })
            
            st.success("FAISS index created and dataset saved successfully.")
            
            # Load the test data
            test_data = pd.read_csv(test_file)
            test_data = test_data.dropna(subset=['Instruction', 'Response'])
            # Define the list of specific instructions to filter
            instructions_of_interest = [
                "What hours am I allowed to work on a Stamp 2 visa?",
                "What courses or programmes can a non-EEA national study in Ireland?",
                "How do I get a PPS Number?",
                "Am I required to have Private Medical Insurance?",
                "What is the total period of time a non-EEA student can be permitted to study in Ireland?"
            ]
            # Load the dataset and FAISS index
            dataset, index = load_dataset(dataset_path, index_file)
            
            # Iterate over the test data to generate and evaluate responses
            rouge_scores_list = []
            # Drop rows with missing 'Instruction' or 'Response'
            test_data = test_data.dropna(subset=['Instruction', 'Response'])

            # Filter the rows based on the specific instructions
            filtered_test_data = test_data[test_data['Instruction'].isin(instructions_of_interest)]

            for idx, row in filtered_test_data.iterrows():
                query = row['Instruction']
                reference_response = row['Response']
                
                if query:
                    # Encode the query
                    query_embedding = encode_text([query], bert_tokenizer, bert_model)
                    k = 1  # Number of nearest neighbors
                    
                    # Perform FAISS search
                    D, I = index.search(query_embedding, k)

                    # Validate if the search returned valid results
                    if len(I) > 0 and len(I[0]) > 0:
                        retrieved_index = I[0][0]
                        
                        # Ensure the index is within bounds
                        if 0 <= retrieved_index < len(train_data):
                            # Retrieve the most similar instruction and response
                            instruction_text = train_data['Instruction'].iloc[retrieved_index]
                            response_text = train_data['Response'].iloc[retrieved_index]

                            st.write(f"Most similar instruction: {instruction_text}")
                            st.write(f"Most similar response: {response_text}")
                            
                            # Generate response from the model
                            generated_text, time_taken = generate_response(gpt_tokenizer, gpt_model, response_text, model_name)
                            st.write("Generated Response: ", generated_text)
                            st.write(f"Time taken to generate response: {time_taken:.2f} seconds")
                            

                        else:
                            st.warning("The index returned by FAISS is out of the bounds of the dataset. Skipping this query.")
                            continue
                    else:
                        st.warning("No similar response found in the dataset. Skipping this query.")
                        continue
                    # Post-process generated text
                    generated_text = post_process_gemma_response(generated_text)
                    rouge_scores = compute_rouge_scores_for_each_generated_response(reference_response, generated_text)
                    st.write("ROUGE Scores for this query:")
                    st.write(format_rouge_scores([rouge_scores]))
                    # Compute and store ROUGE scores
                    rouge_scores_list.append(rouge_scores)
            
            # Display average ROUGE scores
            st.write("Average ROUGE Scores across all responses:")
            if rouge_scores_list:
                avg_rouge_scores = compute_average_rouge_scores_against_test_set(rouge_scores_list)
                formatted_avg_scores = format_rouge_scores(rouge_scores_list)
                st.write(formatted_avg_scores)
                
                # Add average ROUGE scores to performance data
                avg_rouge_data = {
                    'Model': model_name,
                    'Embedding Generation Time (s)': embedding_time,
                    'FAISS Index Creation Time (s)': faiss_creation_time,
                    'ROUGE1 Precision': avg_rouge_scores['rouge1']['precision'],
                    'ROUGE1 Recall': avg_rouge_scores['rouge1']['recall'],
                    'ROUGE1 F1': avg_rouge_scores['rouge1']['fmeasure'],
                    'ROUGE2 Precision': avg_rouge_scores['rouge2']['precision'],
                    'ROUGE2 Recall': avg_rouge_scores['rouge2']['recall'],
                    'ROUGE2 F1': avg_rouge_scores['rouge2']['fmeasure'],
                    'ROUGE-L Precision': avg_rouge_scores['rougeL']['precision'],
                    'ROUGE-L Recall': avg_rouge_scores['rougeL']['recall'],
                    'ROUGE-L F1': avg_rouge_scores['rougeL']['fmeasure']
                }
                performance_data.append(avg_rouge_data)

    # Display comparison table
    if performance_data:
        st.write("Model Performance Comparison")
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df)

if __name__ == "__main__":
    main()
