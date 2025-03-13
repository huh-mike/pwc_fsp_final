import json

def flatten_tagged_data(tagged_data):
    """
    Flattens the tagged data structure into a list of entries.
    Each entry represents either a main article or a PDF.
    """
    flattened_data = []

    for article in tagged_data:
        article_id = article["_id"]

        # Add main article entry
        main_article_entry = {
            "type": "article",
            "id": article_id,
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "text": article.get("text", ""),
            "summary": article.get("summary", ""),
            "tags": article.get("tags", []),
            "embedding_reference": article.get("embedding_reference", "")
        }
        flattened_data.append(main_article_entry)

        # Process PDFs
        if "pdfs" in article and article["pdfs"]:
            for pdf in article["pdfs"]:
                pdf_id = f"{article_id}_pdf"

                # Combine tags from the main article and PDF (avoid duplicates)
                combined_tags = list(set(article.get("tags", []) + pdf.get("pdf_tags", [])))

                # Add PDF entry
                pdf_entry = {
                    "type": "pdf",
                    "id": pdf_id,
                    "title": pdf.get("pdf_title", ""),
                    "url": pdf.get("pdf_url", ""),
                    "text": pdf.get("processed_text", ""),
                    "summary": pdf.get("pdf_summary", ""),
                    "tags": combined_tags,
                    "embedding_reference": pdf.get("pdf_embedding_reference", "")
                }
                flattened_data.append(pdf_entry)

    return flattened_data

# Example usage
if __name__ == "__main__":
    # Path to the tagged data file
    tagged_data_file = "tagged_articles_with_embeddings.json"

    # Load the tagged data
    with open(tagged_data_file, 'r') as f:
        tagged_data = json.load(f)

    # Flatten the data
    flattened_data = flatten_tagged_data(tagged_data)

    # Save the flattened data
    output_file = "flattened_articles_and_pdfs.json"
    with open(output_file, 'w') as f:
        json.dump(flattened_data, f, indent=4)

    print(f"Flattening completed. Updated data saved to {output_file}.")