# My Solution

### Approach

I used a RAG-based solution to return responses to user questions. To create the knowledge base, I made a notebook, `data_preprocessing_batch.ipynb`. This notebook created the knowledge base once - that way it doesn't need to get created every time a query is run. I just apppended the vector embeddings to the CSV table with the responses - `6000_all_categories_questions_with_excerpts_embeddings.csv`.

My general approach was to use the `get_similar_responses` function in `retriever.py` to 1) convert the user question to an embedding, 2) compute the similarity with each of the 6,000 excerpts (I chose not to use an approximate nearest neighbor approach due to the small size of this dataset allowing for fast pairwise calculation), 3) return the closest response. 

In the code, you'll also see my vain attempts to summarize the response in the `generate_answer` function. The idea was to use Google's FLAN-T5 language model to then summarize the returned text based on the user's query. This small model can run locally on my Macbook. The idea was that instead of responding to this question: "What is the population of Maklavan?" with this long answer: "Maklavan: Maklavan (, also Romanized as Mākalān) is a city and capital of Sardar-e Jangal District, in Fuman County, Gilan Province, Iran At the 2006 census, its population was 2,170 individuals", we could instead respond "The population of Maklavan is 2,170." Unfortunately, I couldn't the summarized results returned - only the verbatim text is ever returned. 









# Instructions

### To Install
Create a virtual env
```
python3 -m venv venv
```

Activate the virtual env
```
source ./venv/bin/activate
```

Install the requirements
```
pip install -r requirements.txt
```

### Run the server
```
uvicorn src.main:app --reload
```

To view the swagger documentation navigate to the ipaddress of the server 
```
<IP ADDRESS>:8000/docs
```

If you don't add the /docs part then you will not see the swagger documentation.
