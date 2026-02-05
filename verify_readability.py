import re

def _calculate_readability_score(text: str) -> float:
    """
    Calculate Flesch Reading Ease score.
    206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
    """
    if not text or len(text.strip()) == 0:
        return 0.0
        
    # Clean text (remove HTML)
    clean_text = re.sub(r'<[^>]+>', ' ', text)
    
    # Count sentences
    sentences = len(re.findall(r'[.!?]+', clean_text))
    if sentences == 0: sentences = 1
    
    # Count words
    words_list = re.findall(r'\w+', clean_text)
    words = len(words_list)
    if words == 0: return 0.0
    
    # Count syllables
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count = 1
        return count
        
    syllables = sum(count_syllables(w) for w in words_list)
    
    # Calculate score
    asl = words / sentences
    asw = syllables / words
    
    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return max(0.0, min(100.0, score))

def test_readability():
    # Example of complex text (likely low score)
    complex_text = """
    The fundamental paradigm shift in contemporary technological discourse necessitates a meticulous investigation 
    into the multifaceted complexities of machine learning architectures. It is paramount to observe that 
    the interplay between disparate neural networks underpins the robust framework of artificial intelligence.
    """
    
    # Example of simple text (likely high score)
    simple_text = """
    We need to look at how new technology works. It is important to see how different parts of a computer plan 
    come together. This helps makes the system strong and easy to use.
    """
    
    score_complex = _calculate_readability_score(complex_text)
    score_simple = _calculate_readability_score(simple_text)
    
    print(f"Readability Score (Complex): {score_complex:.2f}")
    print(f"Readability Score (Simple): {score_simple:.2f}")

if __name__ == "__main__":
    test_readability()
