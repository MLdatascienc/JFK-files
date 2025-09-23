import os
import re
import json
import spacy
from collections import defaultdict
import pandas as pd
from textblob import TextBlob
import string
import textstat  # for readability score
from collections import Counter
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Load the spaCy NLP model
nlp = spacy.load("en_core_web_sm")

KEYWORDS = {"CIA", "FBI", "Oswald", "assassination", "Kennedy", "Dallas", "conspiracy", "Ruby", "investigation", "report", "classified", "top secret"}

# Extract text from a .txt file
def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return ""
    return text

# Extract relevant information from the text using spaCy and regex
def extract_interesting_info(text):
    data = defaultdict(set)
    if not text:
        return data

    doc = nlp(text)
    
    # Extract entities
    data["dates"].update(ent.text for ent in doc.ents if ent.label_ == "DATE")
    data["names"].update(ent.text for ent in doc.ents if ent.label_ == "PERSON")
    data["locations"].update(ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC"})
    data["organizations"].update(ent.text for ent in doc.ents if ent.label_ == "ORG")
    
    # Check for keywords
    data["keywords"].update(word for word in text.split() if word in KEYWORDS)
    
    # Find numbers (e.g., phone numbers, dates)
    data["numbers"].update(re.findall(r'\b\d{2,}\b', text))

    return {key: ", ".join(sorted(values)) for key, values in data.items()}

# Extract 'from' and 'to' from the text (looking for sender and recipient keywords)
def extract_sender_recipient(text):
    from_person = None
    to_person = None
    
    # Regex patterns for 'From' and 'To'
    from_patterns = [
        r"From[:\s]+([A-Za-z,.\- ]+)",  # Matches: From: John Doe
        r"Sent by[:\s]+([A-Za-z,.\- ]+)",  # Matches: Sent by: John Doe
        r"Sender[:\s]+([A-Za-z,.\- ]+)",  # Matches: Sender: John Doe
        r"From[:\s]+([A-Za-z,.\- ]+)",  # Matches: From  John Doe
        r"^([A-Za-z,.\- ]+)\s*sent\s*:",  # Matches: John Doe sent:
        r"Dear\s([A-Za-z,.\- ]+)",  # Matches: Dear John Doe
    ]
    
    to_patterns = [
        r"To[:\s]+([A-Za-z,.\- ]+)",  # Matches: To: Jane Doe
        r"Recipient[:\s]+([A-Za-z,.\- ]+)",  # Matches: Recipient: Jane Doe
        r"Dear\s([A-Za-z,.\- ]+)",  # Matches: Dear Jane Doe (in case "Dear" means recipient)
        r"To\s([A-Za-z,.\- ]+)",  # Matches: To John Doe
        r"^([A-Za-z,.\- ]+)\s*received\s*:",  # Matches: John Doe received:
        r"Sincerely,\s([A-Za-z,.\- ]+)",  # Matches: Sincerely, John Doe
    ]

    # Search for the 'from' person using regex
    for pattern in from_patterns:
        match = re.search(pattern, text)
        if match:
            from_person = match.group(1).strip()
            break

    # Search for the 'to' person using regex
    for pattern in to_patterns:
        match = re.search(pattern, text)
        if match:
            to_person = match.group(1).strip()
            break

    return from_person, to_person

# Perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# Extract specific events related to JFK
def extract_jfk_related_events(text):
    events = []
    jfk_keywords = ["Dealey Plaza", "November 22, 1963", "Warren Commission", "Kennedy Assassination", "Dallas"]
    for event in jfk_keywords:
        if event.lower() in text.lower():
            events.append(event)
    return ", ".join(events)

# Get the length of the document in terms of word count
def get_document_length(text):
    return len(text.split())

# Get the most frequent words in the document (excluding stopwords)
def get_most_frequent_words(text, top_n=10):
    stopwords = set(nlp.Defaults.stop_words)
    words = [word for word in text.split() if word.lower() not in stopwords and word not in string.punctuation]
    word_counts = Counter(words)
    return ", ".join([word for word, count in word_counts.most_common(top_n)])

# Calculate text similarity with other documents (Cosine similarity)
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    return cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]

# 1. Add Keyword Frequency Counter
def get_keyword_frequency(text, keywords=KEYWORDS):
    word_counts = {keyword: text.lower().count(keyword.lower()) for keyword in keywords}
    return word_counts

# 2. Sentiment per Section (split by paragraphs)
def analyze_sentiment_per_section(text):
    sections = text.split("\n\n")  # Split into paragraphs or sections
    section_sentiments = {}
    for i, section in enumerate(sections):
        polarity, subjectivity = analyze_sentiment(section)
        section_sentiments[f"Section {i+1}"] = {"polarity": polarity, "subjectivity": subjectivity}
    return section_sentiments

# 3. Add Flesch-Kincaid Readability Score
def get_readability_score(text):
    return textstat.flesch_kincaid_grade(text)

# 4. Add Word Length Analysis (Count long and short words)
def word_length_analysis(text):
    words = text.split()
    short_words = [word for word in words if len(word) <= 4]
    long_words = [word for word in words if len(word) > 7]
    return len(short_words), len(long_words)

# Additional numerical features
def avg_word_length(text):
    words = text.split()
    total_chars = sum(len(word) for word in words)
    return total_chars / len(words) if len(words) > 0 else 0

def avg_sentence_length(text):
    sentences = text.split('.')
    total_words = sum(len(sentence.split()) for sentence in sentences)
    return total_words / len(sentences) if len(sentences) > 0 else 0

def total_syllables(text):
    return textstat.syllable_count(text)

def avg_syllables_per_word(text):
    words = text.split()
    total_syllables_count = sum(textstat.syllable_count(word) for word in words)
    return total_syllables_count / len(words) if len(words) > 0 else 0

def pos_count(text):
    doc = nlp(text)
    pos_counts = {
        "nouns": sum(1 for token in doc if token.pos_ == "NOUN"),
        "verbs": sum(1 for token in doc if token.pos_ == "VERB"),
        "adjectives": sum(1 for token in doc if token.pos_ == "ADJ"),
        "adverbs": sum(1 for token in doc if token.pos_ == "ADV"),
    }
    return pos_counts

def named_entity_freq(text):
    doc = nlp(text)
    entity_counts = {
        "persons": sum(1 for ent in doc.ents if ent.label_ == "PERSON"),
        "organizations": sum(1 for ent in doc.ents if ent.label_ == "ORG"),
        "locations": sum(1 for ent in doc.ents if ent.label_ == "GPE"),
    }
    return entity_counts

def flesch_reading_ease(text):
    return textstat.flesch_reading_ease(text)

def type_token_ratio(text):
    words = text.split()
    unique_words = set(words)
    return len(unique_words) / len(words) if len(words) > 0 else 0

def punctuation_count(text):
    punctuation_marks = string.punctuation
    return {mark: text.count(mark) for mark in punctuation_marks}

# Main processing function for a single .txt file
def process_single_txt(txt_path):
    text = extract_text_from_txt(txt_path)
    extracted_info = {}

    if text:
        extracted_info = extract_interesting_info(text)

        # Extract 'from' and 'to' information
        from_person, to_person = extract_sender_recipient(text)

        # Add new features
        polarity, subjectivity = analyze_sentiment(text)

        # Get frequent words and keyword frequency
        frequent_words = get_most_frequent_words(text)
        keyword_freq = get_keyword_frequency(text)

        # Get readability score
        readability_score = get_readability_score(text)

        # Add extracted data
        extracted_info["TXT File"] = txt_path
        extracted_info["full_text"] = text
        extracted_info["sentiment_polarity"] = polarity
        extracted_info["sentiment_subjectivity"] = subjectivity
        extracted_info["from"] = from_person if from_person else "Unknown"
        extracted_info["to"] = to_person if to_person else "Unknown"
        extracted_info["frequent_words"] = frequent_words
        extracted_info["keyword_frequency"] = json.dumps(keyword_freq)
        extracted_info["readability_score"] = readability_score

        extracted_info["avg_word_length"] = avg_word_length(text)
        extracted_info["avg_sentence_length"] = avg_sentence_length(text)
        extracted_info["total_syllables"] = total_syllables(text)
        extracted_info["avg_syllables_per_word"] = avg_syllables_per_word(text)
        extracted_info["pos_counts"] = json.dumps(pos_count(text))
        extracted_info["named_entity_counts"] = json.dumps(named_entity_freq(text))
        extracted_info["flesch_reading_ease"] = flesch_reading_ease(text)
        extracted_info["type_token_ratio"] = type_token_ratio(text)
        extracted_info["punctuation_counts"] = json.dumps(punctuation_count(text))

    return extracted_info

# Function to save extracted data into JSON and Excel
def save_extracted_data(extracted_data, json_file, excel_file):
    # Save as JSON
    with open(json_file, "w") as f:
        json.dump(extracted_data, f, indent=4)

    # Save as Excel
    df = pd.DataFrame(extracted_data).fillna("")  # Fill missing values with empty strings

    # Write to Excel file
    df.to_excel(excel_file, index=False)

# Function to process .txt files in parallel
def process_txt_files_parallel(txt_directory, json_file, excel_file):
    txt_paths = [os.path.join(txt_directory, filename) for filename in os.listdir(txt_directory) if filename.endswith('.txt')]
    
    # Process txt files using multiprocessing
    with ProcessPoolExecutor() as executor:
        extracted_data = list(executor.map(process_single_txt, txt_paths))

    # Flatten the list of dictionaries and save
    save_extracted_data(extracted_data, json_file, excel_file)

# Add the main guard
if __name__ == '__main__':
    # Specify the .txt directory and output files
    txt_directory = r"C:\Users\u0106491\Documents\LSTAT\Masterthesissen\Voorstellen JFK\2. Transform PDF to text\jfk_2025_texts_reduced"  # Update this path
    output_json = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'extracted_data.json')
    output_excel = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'extracted_data.xlsx')

    # Process the .txt files in parallel
    process_txt_files_parallel(txt_directory, output_json, output_excel)
