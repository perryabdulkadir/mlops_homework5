import os
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model once at module level (more efficient)
model_name = "google/flan-t5-small"  # Small enough to run on MacBook
tokenizer = None
model = None

def load_model_if_needed():
    global tokenizer, model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def string_to_array(embedding_str):
    """Convert string representation of array to numpy array."""
    try:
        # Remove brackets and split by whitespace
        values = embedding_str.strip('[]').split()
        # Convert to float and create numpy array
        return np.array([float(x) for x in values])
    except Exception as e:
        print(f"Error parsing embedding: {e}")
        return np.zeros(384)  # Default dimension for the model

def generate_answer(question, contexts):
    """Generate a concise answer based on retrieved contexts"""
    load_model_if_needed()
    
    # Combine contexts into one text
    context_text = " ".join(contexts)
    
    # Create a more directive prompt
    prompt = f"""Question: {question}

Context: {context_text}

Instructions: Answer the question directly and concisely using only the information from the context. 
Provide just the key facts without unnecessary words.

Answer:"""
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    
    try:
        outputs = model.generate(
            inputs.input_ids, 
            max_length=50,          
            min_length=1,           
            temperature=0.3,         
            num_return_sequences=1,
            do_sample=True,          
            no_repeat_ngram_size=2   
        )
        
        # Decode and return
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the answer if needed
        answer = answer.strip()
        
        # If the answer is too long, truncate it
        if len(answer) > 150:
            answer = answer[:150] + "..."
            
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Could not generate an answer from the context."

def get_similar_responses(question: str, top_k: int = 1) -> dict:
    # Change the default parameter from 5 to 1
    # Rest of the function stays the same
    
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    
    # Use absolute path
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, 'data', '6000_all_categories_questions_with_excerpts_embeddings.csv')
    
    try:
        df = pd.read_csv(csv_path, index_col=0)
        df['embedding'] = df['embedding'].apply(string_to_array)
        
        # Load the embedding model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Step 1: Convert the question to embedding
        question_emb = model.encode([question])[0]

        # Step 2: Compute the similarity 
        excerpt_embs = np.stack(df['embedding'].values)
        similarities = np.dot(excerpt_embs, question_emb) / (
            np.linalg.norm(excerpt_embs, axis=1) * np.linalg.norm(question_emb) + 1e-10
        )

        # Step 3: Prune top k most similar excerpts
        top_k_idx = similarities.argsort()[-top_k:][::-1]
        top_texts = df.iloc[top_k_idx]['wikipedia_excerpt'].tolist()
        
        # Step 5: Generate an answer using the retrieved context
        generated_answer = generate_answer(question, top_texts)
        
        # Return both the answer and supporting texts
        return {
            "generated_answer": generated_answer,
            "supporting_contexts": top_texts
        }
    except Exception as e:
        print(f"Error in retriever: {e}")
        return {
            "generated_answer": "Sorry, I couldn't generate an answer.",
            "supporting_contexts": ["Error processing your request."]
        }
