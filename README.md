# PDF Text Extraction and Analysis

This project is designed to extract, analyze, and process text data from PDF files. The primary goal is to extract relevant content from PDFs using OCR, analyze the extracted text for various features, and then save the results in a structured format (JSON and Excel).

## Features

### 1. **Text Extraction via OCR**
   - The code uses **EasyOCR** to perform Optical Character Recognition (OCR) on images extracted from PDFs. This is particularly useful for scanned PDFs or image-based PDFs where the text is not selectable.

### 2. **Named Entity Recognition (NER)**
   - **spaCy** is used to perform Named Entity Recognition (NER) to extract:
     - **Dates** (e.g., "November 22, 1963")
     - **Names** (e.g., "John Kennedy")
     - **Locations** (e.g., "Dallas")
     - **Organizations** (e.g., "CIA")

### 3. **Keyword Detection**
   - A predefined set of **keywords** related to JFK (such as "CIA", "Oswald", "Kennedy", etc.) is used to search for occurrences in the text.

### 4. **Sender and Recipient Extraction**
   - The code can extract sender and recipient names from the text using regex patterns, useful for analyzing emails or letter-based documents.

### 5. **Sentiment Analysis**
   - Sentiment of the document is analyzed using **TextBlob**, returning the **polarity** (positive or negative sentiment) and **subjectivity** (objective or subjective) of the text.

### 6. **Text Readability**
   - The **Flesch-Kincaid Grade Level** is computed to measure the readability of the document. A higher score indicates a more complex document.
   - Additional readability features include:
     - **Flesch Reading Ease**
     - **Average Word Length**
     - **Average Sentence Length**
     - **Syllables per Word**

### 7. **Word Length Analysis**
   - The document is analyzed for short words (≤ 4 letters) and long words (> 7 letters), providing an understanding of the text's complexity.

### 8. **Most Frequent Words**
   - The most frequent words in the document are extracted, excluding common stopwords.

### 9. **Text Statistics**
   - The document's length in terms of word count is calculated.
   - The **Type-Token Ratio** is computed to measure lexical diversity (the ratio of unique words to total words).

### 10. **Parts of Speech (POS) Analysis**
   - The code counts the occurrences of different parts of speech, including **nouns**, **verbs**, **adjectives**, and **adverbs**.

### 11. **Named Entity Frequency**
   - Counts the frequency of different types of named entities such as **persons**, **organizations**, and **locations**.

### 12. **Punctuation Analysis**
   - The code counts occurrences of various punctuation marks (e.g., period, comma, question mark), useful for stylistic analysis.

### 13. **Document Metadata Extraction**
   - Extracts metadata from the PDF, including information like the **author**, **creation date**, and **modification date**.

### 14. **JFK-related Event Extraction**
   - Searches the document for events related to **JFK** (e.g., "Dealey Plaza", "Warren Commission", "Kennedy Assassination") and extracts them.

### 15. **Document Similarity (Cosine Similarity)**
   - The code compares the text of two documents using **cosine similarity** to detect how similar they are.

---
