from datasets import load_dataset
import json
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def main(json_file, output_file):
    # Load documents from JSON file
    documents = load_documents_from_json(json_file)

    # Summarize the documents
    summaries = summarize_documents(documents)

    # Save the summaries to output file
    with open(output_file, 'w') as file:
        json.dump(summaries, file)


def summarize_documents(documents):
    summaries = []
    for document in documents:
        inputs = tokenizer(document, padding='max_length', truncation=True, max_length=1024, return_tensors="pt")
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=50, num_beams=2, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries


def load_documents_from_json(json_file):
    with open(json_file, 'r') as file:
        documents = json.load(file)
    return documents


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a list of documents in JSON format.")
    parser.add_argument("input_file", help="Path to the input JSON file containing a list of documents.")
    parser.add_argument("output_file", help="Path to the output JSON file to save the summaries.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input_file, args.output_file)

# dataset = load_dataset("cnn_dailymail", version="3.0.0")
# print(f"Features: {dataset['train'].column_names}")


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
