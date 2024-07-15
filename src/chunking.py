import os
import PyPDF2
import re


def extract_sections_from_pdf(pdf_file_path):
    content_by_section = {}
    current_section = None
    current_text = []

    def save_current_section():
        nonlocal current_section, current_text
        if current_section:
            content_by_section[current_section] = "\n".join(current_text).strip()
            current_text = []

    with open(pdf_file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)

        for page in reader.pages:
            text = page.extract_text()

            matches = re.findall(r'(\d+\.?\d*\s+[A-Z][^\n]*)\n', text, flags=re.MULTILINE)

            if matches:
                start = 0
                for match in matches:
                    segment_text = match.strip()
                    start_pos = text.find(match, start)

                    if current_section:
                        content_by_section[current_section] += "\n" + text[start:start_pos].strip()
                    current_section = segment_text
                    if current_section not in content_by_section:
                        content_by_section[current_section] = current_section  # Add section title
                    start = start_pos + len(match)
                if current_section:
                    content_by_section[current_section] += "\n" + text[start:].strip()
            else:
                if current_section:
                    content_by_section[current_section] += "\n" + text.strip()

        save_current_section()

    return content_by_section


def process_pdfs_in_directory(directory_path):
    all_sections = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            sections = extract_sections_from_pdf(pdf_path)
            all_sections.append(sections)
    return all_sections
