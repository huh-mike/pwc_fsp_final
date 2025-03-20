import os
import json
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

from GPTServices import gpt_generate_embedding

USER_QUERY_TOP_N_CONFIG = 2


class RAGSystemCloud:
    def __init__(self):
        """
        Connects to MongoDB Cloud instead of loading local JSON/CSV files.
        """
        MONGO_URI = "mongodb://nehabhakat:kMxFQYPZ5Z5rqHOEVJx2BPToWVyHpmmrpnl9rniu6AokTNMO3TsAhHXeFvQaEv3lNqeTNXctc2EQACDbthoOWw==@nehabhakat.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@nehabhakat@"

        try:
            self.client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True,
                                      serverSelectionTimeoutMS=5000)
            self.db = self.client["FSPDatabase"]
            self.collection = self.db["TaggedCollection"]

            # Check if MongoDB is reachable
            self.client.admin.command('ping')
            print("[✅] Successfully connected to MongoDB Cloud.")

        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise

    def fetch_documents_from_mongo(self, tag_names: List[str]) -> List[str]:
        """
        Fetch documents from MongoDB based on matching tags.

        :param tag_names: List of relevant tag names.
        :return: List of document texts.
        """
        relevant_texts = []
        query = {"processed_data.tags": {"$in": tag_names}}  # Match any tag in the list

        print(f"[🔍] Querying MongoDB with: {query}")

        for doc in self.collection.find(query, {"raw_data.text": 1, "_id": 1, "title": 1}):
            raw_text = doc.get("raw_data", {}).get("text", "")

            # If text is stored as JSON string, parse it
            try:
                parsed_text = json.loads(raw_text).get("content", "").strip() if raw_text.startswith(
                    "{") else raw_text.strip()
            except json.JSONDecodeError:
                parsed_text = raw_text.strip()

            print(
                f"[📄] Found Document: {doc['_id']} - Title: {doc.get('title', 'No Title')} | Text Length: {len(parsed_text)}")

            if parsed_text:
                relevant_texts.append(parsed_text)

        print(f"[✅] Retrieved {len(relevant_texts)} documents from MongoDB.")
        return relevant_texts

    def process_query(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Process the user query and fetch relevant context from MongoDB.

        :param query: User's query string.
        :return: Tuple of (top matching tags, relevant document context).
        """
        try:
            query_embeddings = gpt_generate_embedding(query)
        except Exception as e:
            print(f"[ERROR] Could not generate embedding for query: {e}")
            return [], "Unable to generate query embedding."

        # Get top matching tags
        tag_embeddings = self.load_tag_embeddings_from_mongo()

        if not tag_embeddings:
            print("No tag embeddings found in MongoDB!")
            return [], "No relevant tags found for your query."

        top_n_tags = self.get_top_tags(query_embeddings, tag_embeddings)

        # Fetch relevant documents from MongoDB
        tag_names = [tag['tertiary_category'] for tag in top_n_tags]
        context = "\n\n".join(self.fetch_documents_from_mongo(tag_names))

        if not context.strip():
            context = "No relevant documents found for your query."

        return top_n_tags, context

    def load_tag_embeddings_from_mongo(self) -> Dict[str, List[float]]:
        """
        Load tag embeddings stored in MongoDB.

        :return: Dictionary mapping tags to embedding vectors.
        """
        tag_embeddings = {}
        for doc in self.collection.find({}, {"_id": 0, "processed_data.tags": 1, "processed_data.embedding_vector": 1}):
            tags = doc.get("processed_data", {}).get("tags", [])
            embedding = doc.get("processed_data", {}).get("embedding_vector", [])

            if tags and embedding:
                for tag in tags:
                    tag_embeddings[tag] = embedding  # Store tag and corresponding embedding

        return tag_embeddings

    def get_top_tags(self, query_embeddings, tag_embeddings, top_n=USER_QUERY_TOP_N_CONFIG):
        """
        Computes cosine similarity between query embedding and tag embeddings.

        :return: Top N tags with highest similarity scores.
        """
        similarities = []

        for tag, tag_embedding in tag_embeddings.items():
            similarity = cosine_similarity([query_embeddings], [tag_embedding])[0][0]
            similarities.append((tag, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        return [{'tertiary_category': tag, 'score': score} for tag, score in similarities[:top_n]]


def fetch_relevant_documents(user_query: str) -> str:
    """
    Fetch relevant documents for a given user query using MongoDB-based RAG system.

    :param user_query: The user's query text.
    :return: Context string containing relevant information.
    """
    rag_system = RAGSystemCloud()
    query_top_tags, context = rag_system.process_query(user_query)

    if query_top_tags:
        print("\nUser Query Matched tags:")
        for i, tag in enumerate(query_top_tags, 1):
            print(f"{i}. {tag['tertiary_category']} (Score: {tag['score']:.4f})")
        print(f"\nRetrieved context ({len(context)} characters)")

    if not context.strip():
        return "No relevant documents found. Please refine your query."

    return context
