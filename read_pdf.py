import PyPDF2

with open('CSRT+RCNN.pdf', 'rb') as file:
    pdf = PyPDF2.PdfReader(file)
    for i, page in enumerate(pdf.pages):
        print(f'\n--- Page {i+1} ---\n')
        print(page.extract_text())
