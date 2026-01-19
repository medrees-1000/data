from pypdf import PdfReader

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded file object or a file path.
    """
    try:
        reader = PdfReader(pdf_file)
        pages_text = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                # Cleaning: collapse multiple spaces and newlines
                text = " ".join(text.split())
                pages_text.append(text)
        
        full_text = "\n".join(pages_text)
        
        if len(full_text.strip()) < 50:
            return None
            
        return full_text
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None