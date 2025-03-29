"""
Text utility functions for processing and analyzing conversation text.

This module provides utility functions for text processing, including preprocessing,
similarity calculations, keyword extraction, and other text analysis functions.
"""

import re
import string
from typing import Dict, List, Set, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mcp_therapist.utils.logging import logger

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Initialize NLTK components
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Preprocess text for analysis.
    
    This function:
    1. Converts text to lowercase
    2. Removes punctuation
    3. Removes numbers
    4. Removes stopwords
    5. Lemmatizes words
    
    Args:
        text: The input text to preprocess.
        
    Returns:
        Preprocessed text.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed_tokens = [
        LEMMATIZER.lemmatize(token) for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]
    
    # Join tokens back into a string
    return " ".join(processed_tokens)


def similarity_score(text1: str, text2: str) -> float:
    """Calculate the semantic similarity between two texts.
    
    Args:
        text1: First text for comparison.
        text2: Second text for comparison.
        
    Returns:
        Similarity score between 0 and 1.
    """
    if not text1 or not text2:
        return 0.0
    
    # Use TF-IDF to convert texts to vectors
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
    except ValueError:
        # Handle edge case with empty vocabulary
        return 0.0
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return float(similarity)


# Create an alias for similarity_score for better naming consistency
calculate_similarity = similarity_score


def extract_keywords(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """Extract keywords from text based on TF-IDF scores.
    
    Args:
        text: Input text to extract keywords from.
        top_n: Number of top keywords to return.
        
    Returns:
        List of tuples (keyword, score) sorted by score.
    """
    if not text:
        return []
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Single document TF-IDF extraction
    vectorizer = TfidfVectorizer(max_features=top_n*2)
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_text])
    except ValueError:
        # Handle edge case with empty vocabulary
        return []
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract scores
    scores = zip(feature_names, tfidf_matrix.toarray()[0])
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Return top N
    return sorted_scores[:top_n]


def find_common_phrases(texts: List[str], min_count: int = 2, min_phrase_len: int = 4) -> Dict[str, int]:
    """Find common phrases across multiple texts.
    
    Args:
        texts: List of text strings to analyze.
        min_count: Minimum number of occurrences to include a phrase.
        min_phrase_len: Minimum number of words in a phrase.
        
    Returns:
        Dictionary of {phrase: count}.
    """
    if not texts:
        return {}
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Extract n-grams
    phrases = {}
    
    for text in processed_texts:
        words = text.split()
        
        # Extract phrases of different lengths
        for phrase_len in range(min_phrase_len, min(8, len(words))):
            for i in range(len(words) - phrase_len + 1):
                phrase = " ".join(words[i:i+phrase_len])
                phrases[phrase] = phrases.get(phrase, 0) + 1
    
    # Filter by minimum count
    common_phrases = {phrase: count for phrase, count in phrases.items() if count >= min_count}
    
    # Sort by count
    return dict(sorted(common_phrases.items(), key=lambda x: x[1], reverse=True))


def get_message_sentiment_indicators(text: str) -> Dict[str, float]:
    """Extract basic sentiment indicators from a message.
    
    This is a simple rule-based approach, not a full sentiment analysis.
    
    Args:
        text: The message text to analyze.
        
    Returns:
        Dictionary of sentiment indicators.
    """
    # Define keyword sets
    positive_keywords = {
        'good', 'great', 'excellent', 'happy', 'pleased', 'wonderful', 'fantastic',
        'awesome', 'helpful', 'useful', 'enjoy', 'appreciate', 'thank', 'thanks',
        'love', 'like', 'positive', 'excited', 'interesting', 'beneficial', 'progress'
    }
    
    negative_keywords = {
        'bad', 'terrible', 'awful', 'frustrated', 'confused', 'useless', 'unhelpful',
        'poor', 'disappointed', 'annoying', 'dislike', 'hate', 'negative', 'sad',
        'upset', 'difficult', 'problem', 'issue', 'trouble', 'fail', 'stuck', 'wrong'
    }
    
    uncertainty_indicators = {
        'maybe', 'perhaps', 'not sure', 'uncertain', 'might', 'could be', 'possibly',
        'unclear', 'doubt', 'confused', 'confusing', 'unsure', 'don\'t know', 'idk',
        'ambiguous', 'indecisive'
    }
    
    # Count matches
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    positive_count = len(words.intersection(positive_keywords))
    negative_count = len(words.intersection(negative_keywords))
    
    # Check for uncertainty phrases
    uncertainty_count = sum(1 for phrase in uncertainty_indicators if phrase in text_lower)
    
    # Calculate question count
    question_count = text.count('?')
    
    # Calculate sentiment score (-1 to 1)
    total_sentiment_keywords = positive_count + negative_count
    if total_sentiment_keywords > 0:
        sentiment_score = (positive_count - negative_count) / total_sentiment_keywords
    else:
        sentiment_score = 0.0
    
    return {
        'sentiment_score': sentiment_score,
        'positive_keywords': positive_count,
        'negative_keywords': negative_count,
        'uncertainty_indicators': uncertainty_count,
        'question_count': question_count
    } 