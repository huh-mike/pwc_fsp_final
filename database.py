import json
import os
from pymongo import MongoClient
from bson.objectid import ObjectId

# MongoDB Connection
MONGO_URI = "mongodb://nehabhakat:kMxFQYPZ5Z5rqHOEVJx2BPToWVyHpmmrpnl9rniu6AokTNMO3TsAhHXeFvQaEv3lNqeTNXctc2EQACDbthoOWw==@nehabhakat.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@nehabhakat@"
DB_NAME = "FSPDatabase"
COLLECTION_NAME = "TaggedCollection"

client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# File paths
SCRAPED_DATA_FILE = "/Users/neha/PycharmProjects/pwc_fsp_final/scraped_data.json"
FLATTENED_DATA_FILE = "/Users/neha/PycharmProjects/pwc_fsp_final/flattened_articles_and_pdfs.json"
EMBEDDINGS_FOLDER = "/Users/neha/PycharmProjects/pwc_fsp_final/article_embeddings"  # Ensure this folder contains the embeddings


def load_json(filename):
    """ Load JSON data from a file. """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embedding(file_path):
    """ Load an embedding vector from a given file that contains a list. """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            # Ensure it's a list (correct format)
            if isinstance(data, list):
                return data

            # If it's not a list, return an empty list
            return []

    except (FileNotFoundError, json.JSONDecodeError):
        return []  # Return empty if embedding not found or JSON is invalid



def process_data():
    """ Processes and merges raw data, flattened articles, and embeddings into a unified format. """

    # Load scraped data (raw articles)
    scraped_data = load_json(SCRAPED_DATA_FILE)
    scraped_data_dict = {str(entry.get("_id") or entry.get("id")): entry for entry in scraped_data}  # Map by ID

    # Load flattened articles and PDFs
    flattened_data = load_json(FLATTENED_DATA_FILE)

    final_documents = []

    for entry in flattened_data:
        doc_id = str(entry.get("_id") or entry.get("id")).strip()  # Ensure correct ID reference
        raw_entry = scraped_data_dict.get(doc_id, {})

        # Extract embedding reference
        embedding_filename = entry.get("embedding_reference", None)
        embedding_vector = []

        if embedding_filename:
            embedding_path = os.path.join(EMBEDDINGS_FOLDER, embedding_filename)
            embedding_vector = load_embedding(embedding_path)  # Load actual vector

        # If it's a PDF, iterate through each PDF and store separately
        if "pdfs" in raw_entry and raw_entry["pdfs"]:
            for pdf in raw_entry["pdfs"]:
                pdf_id = f"{doc_id}_{pdf.get('pdf_title', 'unknown')}"  # Unique ID for each PDF
                raw_text = pdf.get("processed_text", "MISSING RAW TEXT").strip()

                formatted_doc = {
                    "_id": pdf_id,  # Use unique ID to ensure all PDFs are stored
                    "type": "pdf",
                    "title": pdf.get("pdf_title", ""),
                    "url": pdf.get("pdf_url", ""),
                    "raw_data": {
                        "text": raw_text  # Store extracted text
                    },
                    "processed_data": {
                        "summary": entry.get("summary", ""),
                        "tags": entry.get("tags", []),
                        "embedding_vector": embedding_vector  # Ensure embeddings are added
                    }
                }
                final_documents.append(formatted_doc)

        # Otherwise, process as a normal article
        else:
            raw_text = raw_entry.get("text", "MISSING RAW TEXT").strip()

            formatted_doc = {
                "_id": ObjectId(doc_id) if ObjectId.is_valid(doc_id) else doc_id,  # Ensure MongoDB _id format
                "type": entry.get("type", "article"),
                "title": entry.get("title", ""),
                "url": entry.get("url", ""),
                "raw_data": {
                    "text": raw_text
                },
                "processed_data": {
                    "summary": entry.get("summary", ""),
                    "tags": entry.get("tags", []),
                    "embedding_vector": embedding_vector  # Ensure embeddings are added
                }
            }

            final_documents.append(formatted_doc)

    return final_documents


def upload_to_mongo():
    """ Uploads processed data to MongoDB without duplicates. """
    documents = process_data()

    for doc in documents:
        result = collection.update_one(
            {"_id": doc["_id"]},
            {"$set": doc},
            upsert=True
        )
        if result.upserted_id:
            print(f"Inserted new document: {doc['_id']}")
        elif result.modified_count:
            print(f"Updated existing document: {doc['_id']}")
        else:
            print(f"No changes for document: {doc['_id']}")

    print(f"✅ Processed {len(documents)} documents successfully.")


if __name__ == "__main__":
    upload_to_mongo()