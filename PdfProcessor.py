import os
import requests
from pdfminer.high_level import extract_text

# Define the folder where PDFs will be saved
PDF_FOLDER = "pdfs"

# Ensure the folder exists
os.makedirs(PDF_FOLDER, exist_ok=True)

def download_pdf(url, filename):
    '''
    :param url: URL to the pdf
    :param filename: The name to save the PDF as (without extension)
    :return: File path to the downloaded pdf
    '''
    response = None
    final_path = os.path.join(PDF_FOLDER, f"{filename}.pdf")

    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(final_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded PDF: {final_path}")
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error Downloading PDF from: {url}, Error: {e}, Status code: {response.status_code if response else 'N/A'}")

    return final_path


def extract_text_from_pdf(pdf_path):
    '''
    :param pdf_path: Filepath of the PDF
    :return: Extracted text
    '''
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}. Error: {e}")
        return ""


