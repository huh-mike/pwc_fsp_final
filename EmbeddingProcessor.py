import ast
import json
import os

from GPTServices import gpt_generate_single_response, gpt_generate_embedding


def generate_embeddings_and_summary_for_articles(input_file: str = 'scraped_data.json',
                                                 output_file: str = 'articles_with_embeddings.json',
                                                 embeddings_dir: str = 'article_embeddings'):
    """
    Generate summaries and embeddings for articles in the database.

    Args:
        input_file: Path to the input JSON file containing article data
        output_file: Path to save the JSON file with summaries and embeddings
        embeddings_dir: Directory to store embedding JSON files
    """
    print(f"Loading articles database from {input_file}...")

    # Create embeddings directory if it doesn't exist
    os.makedirs(embeddings_dir, exist_ok=True)

    # Read the JSON file
    try:
        with open(input_file, 'r') as f:
            articles = json.load(f)

        total_articles = len(articles)
        print(f"Loaded {total_articles} articles from {input_file}")

    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Process each article
    for idx, article in enumerate(articles):
        title = article.get('title', '')
        text = article.get('text', '')


        # Generate summary for article text
        if text and ('summary' not in article or not article['summary']):
            print(f"Generating summary for article {idx + 1}/{total_articles}: {title}")
            article['summary'] = create_article_summary(title, text)

        # Generate embedding for article text
        if text and ('embedding_reference' not in article or not article['embedding_reference']):
            print(f"Generating embedding for article {idx + 1}/{total_articles}")

            # Combine title and summary for embedding
            embedding_text = f"{title}\n\n{article.get('summary', '')}"

            try:
                # Generate embedding
                embedding = gpt_generate_embedding(embedding_text)

                # Save embedding to a JSON file
                embedding_file = f"{embeddings_dir}/article_{idx}.json"
                with open(embedding_file, 'w') as f:
                    json.dump(embedding, f)

                # Store reference to embedding file
                article['embedding_reference'] = f"article_{idx}.json"

                print(f"  → Embedding saved to {embedding_file}")
            except Exception as e:
                print(f"Error generating embedding for article {idx + 1}: {e}")

        # Process PDFs if they exist
        if 'pdfs' in article and article['pdfs']:
            for pdf_idx, pdf in enumerate(article['pdfs']):
                pdf_title = pdf.get('pdf_title', '')
                processed_text = pdf.get('processed_text', '')

                # Generate summary for PDF processed text
                if processed_text and ('pdf_summary' not in pdf or not pdf['pdf_summary']):
                    print(f"Generating summary for PDF {pdf_idx + 1} of article {idx + 1}: {pdf_title}")
                    pdf['pdf_summary'] = create_article_summary(pdf_title, processed_text)

                # Generate embedding for PDF processed text
                if processed_text and ('pdf_embedding_reference' not in pdf or not pdf['pdf_embedding_reference']):
                    print(f"Generating embedding for PDF {pdf_idx + 1} of article {idx + 1}")

                    # Combine title and summary for embedding
                    embedding_text = f"{pdf_title}\n\n{pdf.get('pdf_summary', '')}"

                    try:
                        # Generate embedding
                        embedding = gpt_generate_embedding(embedding_text)

                        # Save embedding to a JSON file
                        embedding_file = f"{embeddings_dir}/article_{idx}_pdf_{pdf_idx}.json"
                        with open(embedding_file, 'w') as f:
                            json.dump(embedding, f)

                        # Store reference to embedding file
                        pdf['pdf_embedding_reference'] = f"article_{idx}_pdf_{pdf_idx}.json"

                        print(f"  → Embedding saved to {embedding_file}")
                    except Exception as e:
                        print(f"Error generating embedding for PDF {pdf_idx + 1} of article {idx + 1}: {e}")

    # Save the updated articles
    try:
        with open(output_file, 'w') as f:
            json.dump(articles, f, indent=2)
        print(f"Successfully saved articles with summaries and embeddings to {output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")

def create_article_summary(title: str, text: str, max_tokens: int = 6000) -> str:
    """
    Create a concise summary of an article using GPT-4o-mini.

    Args:
        title: Article title
        text: Article text content
        max_tokens: Maximum tokens for the summary

    Returns:
        Summarized article content
    """

    # Parse text content if it's stored as a dictionary string
    article_content = text
    if isinstance(text, str):
        try:
            text_dict = ast.literal_eval(text)
            if isinstance(text_dict, dict) and 'content' in text_dict:
                print("[DEBUG] content found, extracting content")
                article_content = text_dict['content']
        except (ValueError, SyntaxError):
            article_content = text

    # Create system prompt for summary generation
    system_prompt = (
        "You are a tax information summarizer. Given an article title and content, "
        "create a concise summary that captures the key tax information. "
        f"Keep your summary under {max_tokens} tokens. Focus on factual information "
        "that would be most relevant for tax-related queries."
    )

    # Construct user prompt with title and content
    user_prompt = f"Title: {title}\n\nContent: {article_content}\n\nPlease summarize this tax article."

    # Generate summary using GPT
    try:
        summary = gpt_generate_single_response(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model="gpt-4o-mini",
            temperature=0.3,
            token_limit=max_tokens
        )
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        # Return a truncated version of the original content as fallback
        return article_content[:6000] + "..."



if __name__ == "__main__":
    generate_embeddings_and_summary_for_articles()