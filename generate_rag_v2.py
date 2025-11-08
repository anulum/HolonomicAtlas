import os
import zipfile
import xml.etree.ElementTree as ET
import re

# Namespace for Word documents
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
BODY_TAG = './/w:body'
PARA_TAG = 'w:p'
RUN_TAG = 'w:r'
TEXT_TAG = 'w:t'

def extract_text_from_docx(docx_path):
    """
    Extracts text from a .docx file using only built-in zipfile/xml
    libraries. No 'pip install' needed.
    """
    text_blocks = []
    try:
        with zipfile.ZipFile(docx_path, 'r') as zf:
            if 'word/document.xml' not in zf.namelist():
                print(f"  [ERROR] 'word/document.xml' not found in {docx_path}")
                return None
                
            with zf.open('word/document.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                
                body = root.find(BODY_TAG, NS)
                if body is None:
                    print(f"  [ERROR] Could not find 'w:body' in {docx_path}")
                    return None
                    
                for para in body.findall(PARA_TAG, NS):
                    para_text = ""
                    for run in para.findall(RUN_TAG, NS):
                        for text_element in run.findall(TEXT_TAG, NS):
                            if text_element.text:
                                para_text += text_element.text
                    if para_text:
                        text_blocks.append(para_text)
                        
        return "\n\n".join(text_blocks)
        
    except Exception as e:
        print(f"  [ERROR] Could not process {docx_path}: {e}")
        return None

def get_rag_filename(paper_name):
    """Creates the P#_L#_Name_RAG.txt filename."""
    if "Paper 0" in paper_name:
        return "P0_Foundational_RAG.txt"
    if "Paper 1 " in paper_name:
        return "P1_L1_Quantum_RAG.txt"
    if "Paper 2" in paper_name:
        return "P2_L2_Neurochemical_RAG.txt"
    if "Paper 3" in paper_name:
        return "P3_L3_Genomic_RAG.txt"
    if "Paper 4" in paper_name:
        return "P4_L4_Synchronization_RAG.txt"
    if "Paper 5" in paper_name:
        return "P5_L5_Psychoemotional_RAG.txt"
    if "Paper 6" in paper_name:
        return "P6_L6_Planetary_RAG.txt"
    if "Paper 7" in paper_name:
        return "P7_L7_Symbolic_RAG.txt"
    if "Paper 8" in paper_name:
        return "P8_L8_Cosmic_RAG.txt"
    if "Paper 9" in paper_name:
        return "P9_L9_Memory_RAG.txt"
    if "Paper 10" in paper_name:
        return "P10_L10_Boundary_RAG.txt"
    if "Paper 11" in paper_name:
        return "P11_L11_Noospheric_RAG.txt"
    if "Paper 12" in paper_name:
        return "P12_L12_Gaian_RAG.txt"
    if "Paper 13" in paper_name:
        return "P13_L13_SourceField_RAG.txt"
    if "Paper 14" in paper_name:
        return "P14_L14_Transdimensional_RAG.txt"
    if "Paper 15" in paper_name:
        return "P15_L15_Consilium_RAG.txt"
    if "Paper 16" in paper_name:
        return "P16_L16_Cybernetic_RAG.txt"

    print(f"  [WARN] Could not create a clean name for {paper_name}. Using default.")
    return os.path.basename(paper_name).replace('.docx', '_RAG.txt')

def process_corpus():
    """
    Scans all /Corpus/ subfolders, finds .docx files, 
    and generates their _RAG.txt counterparts.
    """
    corpus_dir = 'Corpus'
    if not os.path.exists(corpus_dir):
        print(f"Error: '{corpus_dir}' directory not found.")
        return

    print("--- Starting RAG Generation (Zero-Dependency) ---")
    files_created = 0
    
    for root, dirs, files in os.walk(corpus_dir):
        if root == corpus_dir:
            continue

        for file in files:
            if file.endswith('.docx') and not file.startswith('~'):
                docx_path = os.path.join(root, file)
                
                rag_name = get_rag_filename(file)
                rag_path = os.path.join(root, rag_name)
                
                if os.path.exists(rag_path):
                    print(f"  [SKIP] {rag_name} already exists.")
                    continue

                print(f"  [NEW]  Processing {file}...")
                content = extract_text_from_docx(docx_path)
                
                if content:
                    with open(rag_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"    -> Created {rag_name}")
                    files_created += 1

    print("-------------------------------------------------")
    print(f"Done. Created {files_created} new _RAG.txt files.")

if __name__ == "__main__":
    process_corpus()