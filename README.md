# My Solution

### Approach

I used a RAG-based solution to return responses to user questions. To create the knowledge base, I made a notebook, `data_preprocessing_batch.ipynb`. This notebook created the knowledge base once - that way it doesn't need to get created every time a query is run. I just apppended the vector embeddings to the CSV table with the responses - `6000_all_categories_questions_with_excerpts_embeddings.csv`.

My general approach was to use the `get_similar_responses` function in `retriever.py` to 1) convert the user question to an embedding, 2) compute the similarity with each of the 6,000 excerpts (I chose not to use an approximate nearest neighbor approach due to the small size of this dataset allowing for fast pairwise calculation), 3) return the closest response. 

After collecting the verbatim responses (referred to as "sources" in the response), I use the `generate_answer` function to explicitly answer the user's question. I used Google's FLAN-T5 language model to summarize the returned text based on the user's query. This small model can run locally on my Macbook. Here are some examples from testing in Swagger docs. 

When given the question, "What is the population of Maklavan?", the closest excerpt is this:  "Maklavan: Maklavan (, also Romanized as Mākalān) is a city and capital of Sardar-e Jangal District, in Fuman County, Gilan Province, Iran At the 2006 census, its population was 2,170 individuals". The language model allows us to return the much more concise "2,170" to the user (the "answer"). 

#### Query

![Alt text](/screenshots/q1_query.png?raw=true)

#### Response
![Alt text](/screenshots/q1_response.png?raw=true)


Another example, this time properly returning the year only when asked: 
#### Query
![Alt text](/screenshots/q2_query.png?raw=true)


#### Response
![Alt text](/screenshots/q2_response.png?raw=true)

The system is limited; it will always return a response, even if the answer to the question isn't in the database. 
#### Query
![Alt text](/screenshots/q3_query.png?raw=true)


#### Response
![Alt text](/screenshots/q3_response.png?raw=true)



# Testing
I added several unit tests for the retriever and API. 

![Alt text](/screenshots/test_1.png?raw=true)

![Alt text](/screenshots/test_2.png?raw=true)

![Alt text](/screenshots/test_3.png?raw=true)


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
