import time
import json
from typing import List, Dict
from pydantic import BaseModel, ValidationError, Field
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------------------------------------
# 1. Existing Pydantic Model + BEQA Transform Code (Groq)
# -------------------------------------------------------------------
class QAItem(BaseModel):
    Question: str = Field(..., description="The transformed question")
    Answer: str = Field(..., description="The transformed answer")

THROTTLE_LIMIT = 3  # Seconds between consecutive API calls
MAX_RETRIES = 5     # Maximum number of retries for valid responses

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Example endpoint

def parse_groq_output(raw_output: str) -> List[Dict]:
    """
    Parse and validate the output from the Groq model.
    """
    try:
        data = json.loads(raw_output)
        if not isinstance(data, list):
            print("Parsed data is not a list.")
            return []
        items = []
        for item in data:
            qa_item = QAItem(**item)  # Validate via Pydantic
            items.append(qa_item.dict())
        return items

    except (ValidationError, json.JSONDecodeError, TypeError) as e:
        print(f"Failed to parse or validate JSON output: {e}")
        return []

def transform_with_groq(context: str, original_query: str, model_id: str, api_key: str) -> List[Dict]:
    """
    Transform retrieved context into structured QA pairs using the Groq API with retries.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    prompt_template = f"""
You are given:
- Context: "{context}"
- Original Query: "{original_query}"

Your task:
1. Generate one or more question-answer pairs that capture key information from the context.
2. Output your result as valid JSON (no markdown) and use the following schema (an array of objects):
   [
     {{
       "Question": "...",
       "Answer": "..."
     }},
     ...
   ]

Constraints:
- Make sure the JSON is valid.
- No additional keys beyond 'Question' and 'Answer'.
- Do not wrap the output in quotes or markdown.
- If you are unsure, keep it simple.
"""

    retries = 0
    while retries < MAX_RETRIES:
        try:
            data = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_template.strip()
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            response = requests.post(GROQ_API_URL, headers=headers, json=data)
            response.raise_for_status()

            raw_output = response.json()['choices'][0]['message']['content'].strip()
            qa_items = parse_groq_output(raw_output)
            if qa_items:
                return qa_items
            else:
                print(f"[Attempt {retries+1}] Model output was not valid JSON or did not match schema. Retrying...")

        except (requests.exceptions.RequestException, ValidationError, json.JSONDecodeError) as e:
            print(f"[Attempt {retries+1}] Error: {e}. Retrying...")

        # Increment retries and throttle
        retries += 1
        time.sleep(THROTTLE_LIMIT)

    print("Maximum retries reached. No valid response obtained.")
    return []

# -------------------------------------------------------------------
# 2. Sample Data Builders
# -------------------------------------------------------------------
def build_queries_and_corpus(cleaned_data: List[dict]):
    """
    For each item in cleaned_data, we read 'Question' & 'Context' to build two lists:
      - queries (List[str])
      - corpus (List[str])
    """
    queries = []
    corpus = []
    for item in cleaned_data:
        q = item.get("Question") or ""
        c = item.get("Context") or ""
        queries.append(q)
        corpus.append(c)
    return queries, corpus


def vanilla_tfidf_retrieval(queries: List[str], corpus: List[str]):
    """
    Build a TF-IDF vectorizer on the corpus, transform each query, 
    and compute the maximum similarity score for each query.
    """
    vectorizer = TfidfVectorizer()
    corpus_tfidf = vectorizer.fit_transform(corpus)  # shape=(#docs, #features)

    # For demonstration, retrieve only the top doc for each query
    # Also store which doc is top for each query
    sim_scores_list = []
    best_doc_indices = []

    for q in queries:
        query_vec = vectorizer.transform([q])  # shape=(1, #features)
        scores = query_vec.dot(corpus_tfidf.T).toarray().flatten()
        max_idx = int(np.argmax(scores))
        max_score = float(scores[max_idx])
        sim_scores_list.append(max_score)
        best_doc_indices.append(max_idx)

    return sim_scores_list, best_doc_indices

# -------------------------------------------------------------------
# 3. Integration: Use TF-IDF, Then BEQA, Then Similarity
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Suppose 'full_data' is a dictionary with dataset_name -> List[dict], as in your code
    # Each dict has { "Question": ..., "Context": ..., "Response": ... }

    # Example (simplified)
    full_data = {
        "squad": [
            {"Question": "When was Python created?", 
             "Context": "Python is a language created by Guido van Rossum, first released in 1991.",
             "Response": "1991"},
            {"Question": "What does GIL stand for?",
             "Context": "Python's GIL stands for Global Interpreter Lock.",
             "Response": "Global Interpreter Lock"}
        ],
        "wiki_qa": [
            {"Question": "Who developed Python?",
             "Context": "Guido van Rossum developed Python.", 
             "Response": "Guido van Rossum"}
        ]
    }

    # Define your Groq credentials
    API_KEY = "<YOUR_KEY>"
    MODEL_ID = "llama-3.3-70b-versatile"

    for dataset_name, dataset_content in full_data.items():
        print(f"\n=== Processing dataset: {dataset_name} ===")
        queries, corpus = build_queries_and_corpus(dataset_content)

        # Step A: Use TF-IDF to retrieve top doc for each query
        similarities, best_doc_indices = vanilla_tfidf_retrieval(queries, corpus)
        print("Baseline TF-IDF Similarities:", similarities)

        # Step B: Apply BEQA transform to the retrieved doc for each query
        #         Then optionally compute the similarity with the QA pairs
        #         For demonstration, let's just transform the doc -> QA pairs,
        #         and compute a new similarity using the QA pairs' question text.

        second_stage_similarities = []
        for i, q in enumerate(queries):
            top_doc = corpus[best_doc_indices[i]]

            # 1) Transform the doc into QA pairs
            beqa_result = transform_with_groq(
                context=top_doc,
                original_query=q,
                model_id=MODEL_ID,
                api_key=API_KEY
            )

            # 2) (Optional) Build a 'mini-corpus' from the QA items' questions/answers
            #    Let's pick just the 'Question' fields for a new corpus
            beqa_questions = [qa["Question"] for qa in beqa_result]
            # If there are no QA pairs, skip
            if not beqa_questions:
                second_stage_similarities.append(0.0)
                continue

            # 3) Use TF-IDF again, but now on the QA "questions" from the model
            mini_vectorizer = TfidfVectorizer()
            mini_corpus_tfidf = mini_vectorizer.fit_transform(beqa_questions)
            query_vec = mini_vectorizer.transform([q])
            scores = query_vec.dot(mini_corpus_tfidf.T).toarray().flatten()
            best_score = float(np.max(scores))
            second_stage_similarities.append(best_score)

        print("BEQA-based Similarities:", second_stage_similarities)

        # If you want, you could also measure similarity to the "Answer" fields 
        # or combine them. This is flexible depending on your use case.

        print ("Ratio of Improvement:", ratio_of_improvement(second_stage_similarities, similarities))
