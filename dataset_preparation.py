from datasets import load_dataset
from bs4 import BeautifulSoup
from random import choice

def clean_html(html_content: str) -> str:
    """Clean HTML content by removing HTML tags."""
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(strip=True)
    return ""

def clean_description(description: list) -> str:
    """Clean description list into a single string."""
    if description:
        return " ".join(description)  # Join list items into a single string
    return ""

def stream_clean_rag_samples(dataset_name: str, dataset_config: str, num_samples: int):

    #placeholder for the data
    data = []

    # Load the dataset from Hugging Face with streaming enabled and a specific configuration
    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)  # Adjust the split if necessary

    # Stream random samples
    count = 0
    for sample in dataset:
        if count >= num_samples:
            break

        # Extract question (handling 'text' subfield if present)
        question = sample.get("question", None)
        if isinstance(question, dict):
            question = question.get("text", None)

        context = sample.get("document_title", None) or sample.get("context", None) or sample.get("document", None) or sample.get("search_results", None)

        # Clean HTML or description if the context contains the relevant subfields
        if context and isinstance(context, dict):
            if "html" in context:
                context = clean_html(context["html"])  # Clean the HTML content to plain text
            elif "description" in context:
                context = clean_description(context["description"])  # Convert the list to a plain string

        # Handle response, checking for different possible field names
        try:
            response = choice(sample.get("answer", {}).get("aliases", [None])) or sample.get("long_answer_candidates", None) or sample.get("answers", {}).get("text", [None])[0]

            # If response is a list or dict, extract the answer text
            if isinstance(response, list):
              response = response[0] if response else None  # Take the first answer if it's a list
            elif isinstance(response, dict):
              response = response.get("text", None) or response.get("value", None) # Get value from 'text' or 'value' field

        except:
            response = sample.get("answer", None) or sample.get("long_answer_candidates", None) or sample.get("answers", {}).get("text", [None])[0]


        # Check if any field is missing and print the actual field names in the sample
        missing_fields = []
        if question is None:
            missing_fields.append("question")
        if context is None:
            missing_fields.append("context/document/search_results")
        if response is None:
            missing_fields.append("response/long_answer_candidates/answer")

        if missing_fields:
            # Print the actual field names present in the sample
            print(f"Sample {count + 1} - Present fields: {', '.join(sample.keys())}")

        # Clean output
        #print(f"Sample {count + 1}:")
        #print(f"Question: {question if question else 'No question found.'}")
        #print(f"Context: {context if context else 'No context found.'}")
        #print(f"Response: {response if response else 'No response found.'}")
        #print("-" * 50)

        data.append({"Question": question, "Context": context, "Response": response})

        count += 1

    return data

# Example usage: Streaming from SQuAD, Trivia QA, and WikiQA datasets
datasets = [
    ("squad", None),  # SQuAD dataset
    ("trivia_qa", "rc"),  # Config is required for trivia_qa
    ("wiki_qa", None)  # WikiQA dataset
]

full_data = {}

for dataset_name, dataset_config in datasets:
    print(f"Streaming from dataset: {dataset_name}")
    full_data[dataset_name] = stream_clean_rag_samples(dataset_name, dataset_config, num_samples=5)
    print("=" * 100)
