import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# Load taxonomy data and embeddings
def load_taxonomy_embeddings(taxonomy_csv, taxo_embeddings_dir):
    """
    Loads the taxonomy CSV and corresponding embeddings.
    Returns a dictionary mapping tertiary category names to their embeddings.
    """
    taxonomy_df = pd.read_csv(taxonomy_csv)
    tag_embeddings = {}

    for _, row in taxonomy_df.iterrows():
        # Use only the Tertiary Category as the tag
        tag_name = row['Tertiary Category']
        embedding_file = row['embedding_reference']
        embedding_path = os.path.join(taxo_embeddings_dir, embedding_file)

        try:
            with open(embedding_path, 'r') as f:
                embedding = json.load(f)
            
            tag_embeddings[tag_name] = np.array(embedding)
        except FileNotFoundError:
            print(f"Error: Embedding file not found for tag '{tag_name}' at path {embedding_path}. Skipping...")

    return tag_embeddings

# Load article or PDF embeddings
def load_article_embeddings(article_data, article_embeddings_dir):
    """
    Loads the embeddings for articles or PDFs.
    Returns a dictionary mapping article IDs to their embeddings.
    """
    article_embeddings = {}

    for article in article_data:
        article_id = article["_id"]
        
        # Load main article embedding
        embedding_reference = article.get("embedding_reference")
        if embedding_reference:
            embedding_path = os.path.join(article_embeddings_dir, embedding_reference)
            try:
                with open(embedding_path, 'r') as f:
                    embedding = json.load(f)
                
                article_embeddings[article_id] = np.array(embedding)
            except FileNotFoundError:
                print(f"Error: Embedding file not found for article ID {article_id} at path {embedding_path}. Skipping...")
        
        # Load PDF embeddings
        if "pdfs" in article and article["pdfs"]:
            for pdf in article["pdfs"]:
                pdf_id = f"{article_id}_pdf"
                pdf_embedding_reference = pdf.get("pdf_embedding_reference")
                
                if pdf_embedding_reference:
                    pdf_embedding_path = os.path.join(article_embeddings_dir, pdf_embedding_reference)
                    try:
                        with open(pdf_embedding_path, 'r') as f:
                            pdf_embedding = json.load(f)
                        
                        article_embeddings[pdf_id] = np.array(pdf_embedding)
                    except FileNotFoundError:
                        print(f"Error: Embedding file not found for PDF in article ID {article_id} at path {pdf_embedding_path}. Skipping...")

    return article_embeddings

# Compute cosine similarity and get top 5 tags
def get_top_tags(article_embedding, tag_embeddings, top_n=5):
    """
    Computes cosine similarity between an article/PDF embedding and all tag embeddings.
    Returns the top N tags with the highest similarity scores.
    """
    similarities = []

    for tag, tag_embedding in tag_embeddings.items():
        similarity = cosine_similarity([article_embedding], [tag_embedding])[0][0]
        similarities.append((tag, similarity))
    
    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Extract top N tags
    top_tags = [tag for tag, _ in similarities[:top_n]]
    return top_tags

# Main function to tag articles and PDFs
def tag_articles_and_pdfs(taxonomy_csv, taxo_embeddings_dir, article_data, article_embeddings_dir):
    """
    Tags articles and PDFs with the top 5 most relevant tags based on cosine similarity.
    """
    # Load taxonomy embeddings
    tag_embeddings = load_taxonomy_embeddings(taxonomy_csv, taxo_embeddings_dir)

    # Load article embeddings
    article_embeddings = load_article_embeddings(article_data, article_embeddings_dir)

    # Tag each article/PDF
    tagged_data = []

    for article in article_data:
        article_id = article["_id"]

        # Tag main article
        if article_id in article_embeddings:
            article_embedding = article_embeddings[article_id]
            top_tags = get_top_tags(article_embedding, tag_embeddings)
            article["tags"] = top_tags
        else:
            print(f"Skipping article ID {article_id} due to missing embedding.")

        # Tag PDFs
        if "pdfs" in article and article["pdfs"]:
            for pdf in article["pdfs"]:
                pdf_id = f"{article_id}_pdf"
                
                if pdf_id in article_embeddings:
                    pdf_embedding = article_embeddings[pdf_id]
                    top_tags = get_top_tags(pdf_embedding, tag_embeddings)
                    pdf["pdf_tags"] = top_tags
                else:
                    print(f"Skipping PDF in article ID {article_id} due to missing embedding.")

        tagged_data.append(article)

    return tagged_data

# Example usage
if __name__ == "__main__":
    # Paths to input files and directories
    taxonomy_csv = "taxonomy_with_embeddings.csv"
    taxo_embeddings_dir = "taxo_embeddings"
    article_data_file = "articles_with_embeddings.json"
    article_embeddings_dir = "article_embeddings"

    # Load article data
    with open(article_data_file, 'r') as f:
        article_data = json.load(f)

    # Tag articles and PDFs
    tagged_data = tag_articles_and_pdfs(taxonomy_csv, taxo_embeddings_dir, article_data, article_embeddings_dir)

    # Save the updated data with tags
    output_file = "tagged_articles_with_embeddings.json"
    with open(output_file, 'w') as f:
        json.dump(tagged_data, f, indent=4)

    print(f"Tagging completed. Updated data saved to {output_file}.")