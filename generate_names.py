import json
import string

def is_pronounceable(name):
    vowels = set('aeiouy')
    
    # Vowel distribution filter: require 1-2 vowels
    vowel_count = sum(1 for char in name if char in vowels)
    if vowel_count < 1 or vowel_count > 3:
        return False
    
    # Pattern filter: no 3+ consecutive same letters
    for i in range(len(name) - 2):
        if name[i] == name[i+1] == name[i+2]:
            return False
    
    # Pattern filter: avoid too many repeated characters
    char_counts = {}
    for char in name:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # No character should appear more than 2 times
    if any(count > 2 for count in char_counts.values()):
        return False
    
    return True

def generate_names():
    alphabet = list(string.ascii_lowercase)
    excluded_chars = ['x', 'z']
    
    allowed_chars = [char for char in alphabet if char not in excluded_chars]
    
    names = []
    
    for c1 in allowed_chars:
        for c2 in allowed_chars:
            for c3 in allowed_chars:
                for c4 in allowed_chars:
                    for c5 in allowed_chars:
                        name = c1 + c2 + c3 + c4 + c5
                        unique_chars = set(name)
                        if len(unique_chars) >= 4 and is_pronounceable(name):
                            names.append(name)
    
    with open('names.json', 'w') as f:
        json.dump(names, f, indent=2)
    
    print(f"Generated {len(names)} names and saved to names.json")
    print(f"Using characters: {', '.join(allowed_chars)}")

if __name__ == "__main__":
    generate_names()