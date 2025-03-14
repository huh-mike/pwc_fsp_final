import numpy as np
import pandas as pd
import ast
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

from GPTServices import gpt_generate_embedding

USER_QUERY_TOP_N_CONFIG = 2

class RAGSystem:
    def __init__(self):
        """
        Using: flattened_articles_and_pdfs.json, Taxonomy_with_embeddings.csv, (folder)taxo_embeddings
        """
        self.taxonomy_file = "Taxonomy_with_embeddings.csv"
        self.flattened_data_file = "flattened_articles_and_pdfs.json"
        self.taxo_embeddings_folder = "taxo_embeddings"
        
        # Load taxonomy data with embeddings
        self.taxonomy_df = pd.read_csv(self.taxonomy_file)
        
        # Load tag embeddings
        self.tag_embeddings = self._load_tag_embeddings()
        
        # Load flattened articles and PDFs
        with open(self.flattened_data_file, 'r') as f:
            self.flattened_data = json.load(f)

    def _load_tag_embeddings(self) -> Dict[str, List[float]]:
        """
        Load all taxonomy embeddings from the taxo_embeddings folder
        
        :return: Dictionary of tag names to embedding vectors
        """
        tag_embeddings = {}
        
        for _, row in self.taxonomy_df.iterrows():
            tertiary_category = row['Tertiary Category']
            embedding_reference = row['embedding_reference']
            
            # Load the embedding from the referenced file
            embedding_path = os.path.join(self.taxo_embeddings_folder, embedding_reference)
            
            try:
                with open(embedding_path, 'r') as f:
                    embedding = json.load(f)
                    tag_embeddings[tertiary_category] = embedding
            except Exception as e:
                print(f"[WARNING] Could not load embedding for {tertiary_category}: {e}")
                
        return tag_embeddings

    def process_query(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Helper Function to compile the tags and context retrieved

        :param query: User query string
        :return: Tuple of (top matching tags, context from relevant documents)
        """
        try:
            query_embeddings = gpt_generate_embedding(query)
        except Exception as e:
            print(f"[ERROR] Could not generate embedding for user query: {e}")
            raise

        # Get top matching tags based on cosine similarity
        top_n_tags = self.get_top_tags(query_embeddings, self.tag_embeddings)
        
        # Extract relevant documents based on matched tags
        context = self._fetch_context_from_tags(top_n_tags)

        return top_n_tags, context
    
    def _fetch_context_from_tags(self, top_n_tags: List[Dict[str, Any]]) -> str:
        """
        Fetch relevant context from flattened data based on matching tags
        
        :param top_n_tags: List of top matching tags
        :return: Concatenated context string from relevant documents
        """
        tag_names = [tag['tertiary_category'] for tag in top_n_tags]
        relevant_texts = []
        
        # Search through flattened data to find documents with matching tags
        for item in self.flattened_data:
            if 'tags' in item:
                # Check if any of the top tags match this document's tags
                if any(tag in item['tags'] for tag in tag_names):
                    # Extract text content
                    if 'text' in item:
                        # If the text is in JSON string format, parse it
                        if isinstance(item['text'], str) and item['text'].startswith('{'):
                            try:
                                text_data = ast.literal_eval(item['text'])
                                if 'content' in text_data:
                                    relevant_texts.append(text_data['content'])
                            except:
                                relevant_texts.append(item['text'])
                        else:
                            relevant_texts.append(item['text'])
        
        # Combine all relevant texts
        context = "\n\n".join(relevant_texts)
        return context

    def get_top_tags(self, query_embeddings, tag_embeddings, top_n=USER_QUERY_TOP_N_CONFIG):
        """
        Computes cosine similarity between an article/PDF embedding and all tag embeddings.
        Returns the top N tags with the highest similarity scores.
        """
        similarities = []

        for tag, tag_embedding in tag_embeddings.items():
            similarity = cosine_similarity([query_embeddings], [tag_embedding])[0][0]
            similarities.append((tag, similarity))

        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Extract top N tags with their scores
        top_tags = [
            {'tertiary_category': tag, 'score': score} 
            for tag, score in similarities[:top_n]
        ]
        
        return top_tags


def fetch_relevant_documents(user_query: str) -> str:
    """
    Fetch relevant documents for a given user query using the RAG system.

    Args:
        user_query: The user's query text

    Returns:
        Context string containing relevant information based on matched tags
    """
    rag_system = RAGSystem()
    query_top_tags, context = rag_system.process_query(user_query)

    # Print matched tags for debugging
    if query_top_tags:
        print("\nUser Query Matched tags:")
        for i, tag in enumerate(query_top_tags, 1):
            print(f"{i}. {tag['tertiary_category']} (Score: {tag['score']:.4f})")

        print(f"\nRetrieved context ({len(context)} characters)")

    return context



