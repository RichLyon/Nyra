import openai
from dotenv import load_dotenv
import os
import json
from textblob import TextBlob
import spacy
import uuid

# Initial personality traits
personality = {
    "openness": "high",
    "conscientiousness": "moderate",
    "agreeableness": "high",
    "extraversion": "low",
    "neuroticism": "low",
    "emotional_intelligence": "high",
    "humor": "witty",
    "empathy": "very high",
    "curiosity": "moderate",
    "optimism": "high",
    "formality": "moderate",
    "creativity": "high",
    "patience": "very high"
}

# Function to save personality to a file
def save_personality(personality, filename="nyra_personality.json"):
    with open(filename, "w") as f:
        json.dump(personality, f)

# Function to load personality from a file
def load_personality(filename="nyra_personality.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return personality  # Return initial personality if file doesn't exist

def update_personality(personality, trait, change):
    levels = {
        "very low": 1,
        "low": 3,
        "moderate": 5,
        "high": 7,
        "very high": 9
    }
    reverse_levels = {v: k for k, v in levels.items()}
    
    if trait in personality:
        current_level = levels[personality[trait]]
        new_level = max(1, min(9, current_level + change))
        personality[trait] = reverse_levels[new_level]
    
    return personality

user_id_file = "user_id.txt"

# Function to generate a unique user ID
def generate_user_id():
    return str(uuid.uuid4())

# Function to save user ID to a file
def save_user_id(user_id, filename=user_id_file):
    with open(filename, "w") as f:
        f.write(user_id)

# Function to load user ID from a file
def load_user_id(filename=user_id_file):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return f.read().strip()
    return None

# Function to load user profile based on user ID
def load_user_profile(user_id=None, profile_filename="nyra_user_profiles.json"):
    profiles = {}
    
    # Check if the profile file exists, if so, load it
    if os.path.exists(profile_filename):
        with open(profile_filename, "r") as f:
            profiles = json.load(f)
    
    if user_id is None or user_id not in profiles:
        # Generate a new user ID if not provided or not found in existing profiles
        user_id = generate_user_id()
        # Ask the user for their name only when creating a new profile
        user_name = input("Nyra: Hi there! What should I call you? ")
        # Create a new user profile with Nyra's initial personality preferences and user name
        profiles[user_id] = {
            "name": user_name,
            "preferences": personality,
            "history": []
        }
        save_user_profile(user_id, profiles[user_id], profile_filename)
        # Save the generated user ID to a file
        save_user_id(user_id)
    
    return profiles.get(user_id), user_id


# Function to save user profile based on user ID
def save_user_profile(user_id, profile, filename="nyra_user_profiles.json"):
    profiles = {}
    
    # Load existing profiles if the file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            profiles = json.load(f)
    
    # Update the specific user profile
    profiles[user_id] = profile
    
    # Write back the profiles to the JSON file
    with open(filename, "w") as f:
        json.dump(profiles, f, indent=4)

# Memory handling functions
def load_memory(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []  # Return an empty list if the file doesn't exist

def save_memory(memory, filename):
    with open(filename, "w") as f:
        json.dump(memory, f)

def add_memory(memory, interaction, memory_type="short-term"):

    # Extract keywords, sentiment, and category
    interaction["keywords"] = extract_keywords(interaction["user_input"])
    interaction["sentiment"] = analyze_sentiment(interaction["user_input"])
    interaction["category"] = categorize_topics(interaction["user_input"])
    
    if memory_type in memory and isinstance(memory[memory_type], list):
        memory[memory_type].append(interaction)
    else:
        print("Error: Invalid memory type or structure")
    
    save_memory(memory)

def manage_memory(memory, interaction):

    # Extract and store metadata before saving to memory
    interaction["keywords"] = extract_keywords(interaction["user_input"])
    interaction["sentiment"] = analyze_sentiment(interaction["user_input"])
    interaction["category"] = categorize_topics(interaction["user_input"])
    
    if should_store_long_term(interaction["user_input"]):
        memory["long-term"].append(interaction)
        save_memory(memory["long-term"], filename="nyra_memory_long_term.json")
    else:
        memory["short-term"].append(interaction)
        memory["short-term"] = prioritize_memory(memory["short-term"])
        save_memory(memory["short-term"], filename="nyra_memory_short_term.json")

# Keyword and sentiment analysis
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return keywords

nlp = spacy.load("en_core_web_sm")

def categorize_topics(text):
    doc = nlp(text)
    topics = []

    # Extract named entities as topics
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "PERSON", "NORP", "FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"]:
            topics.append(ent.text)

    # Keyword-based topic categorization
    keywords = {
        "technology": ["tech", "software", "hardware", "AI", "machine learning", "robotics"],
        "health": ["medicine", "health", "disease", "treatment", "doctor", "hospital"],
        "finance": ["money", "finance", "investment", "stocks", "bank", "cryptocurrency"],
        "education": ["school", "university", "education", "learning", "teaching"],
        "sports": ["sports", "soccer", "basketball", "baseball", "tennis", "athlete"],
        "entertainment": ["movie", "music", "concert", "celebrity", "TV show", "game"],
        "politics": ["politics", "election", "government", "law", "policy", "senate"],
        "science": ["science", "research", "experiment", "biology", "chemistry", "physics"],
        "space": ["space", "planet", "galaxy", "NASA", "astronomy", "cosmos"],
    }

    for category, words in keywords.items():
        if any(word in text.lower() for word in words):
            topics.append(category)

    # Extract noun chunks as potential topics
    for chunk in doc.noun_chunks:
        topics.append(chunk.text)

    # Remove duplicates and return topics
    unique_topics = list(set(topics))

    return unique_topics if unique_topics else ["general"]

def analyze_sentiment(text):
    # Initialize TextBlob object
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Define thresholds for nuanced sentiment analysis
    if polarity > 0.5:
        return "very positive"
    elif 0.2 < polarity <= 0.5:
        return "positive"
    elif 0.1 < polarity <= 0.2:
        return "slightly positive"
    elif -0.1 <= polarity <= 0.1:
        return "neutral"
    elif -0.2 <= polarity < -0.1:
        return "slightly negative"
    elif -0.5 <= polarity < -0.2:
        return "negative"
    else:
        return "very negative"

# Function to recall relevant memory based on current input
def recall_relevant_memory(memory_segment, current_input):
    relevant_memories = []
    keywords = extract_keywords(current_input)

    print(f"DEBUG: Extracted keywords: {keywords}")  # Debugging statement

    for interaction in memory_segment:
        if isinstance(interaction, dict):  # Check if interaction is a dictionary
            interaction_keywords = interaction.get("keywords", [])
            if any(keyword in interaction_keywords for keyword in keywords):
                relevant_memories.append(interaction)
                print(f"DEBUG: Relevant memory found: {interaction}")  # Debugging statement
        else:
            print(f"Warning: Encountered a non-dictionary interaction: {interaction}")
    
    if relevant_memories:
        return "In previous conversations, you mentioned: " + \
               "; ".join([f"{m['user_input']} ({m['sentiment']})" for m in relevant_memories])
    return ""

# Function to decide if an interaction should be stored long-term
def should_store_long_term(user_input):

    # Analyze sentiment
    sentiment = analyze_sentiment(user_input)
    
    # Keywords that may indicate the importance of the interaction
    important_keywords = ["important", "urgent", "critical", "must remember", "don't forget", "note this"]
    
    # Categories that might be of long-term interest
    important_categories = ["health", "finance", "personal information", "security", "career", "education"]
    
    # Check for keywords indicating importance
    if any(keyword in user_input.lower() for keyword in important_keywords):
        return True
    
    # Check for strong sentiment
    if sentiment in ["very positive", "very negative"]:
        return True
    
    # Categorize the input and check if it matches any important category
    categories = categorize_topics(user_input)
    if any(category in important_categories for category in categories):
        return True
    
    return False


def prioritize_memory(memory, limit=100):

    # Ensure all items in memory are dictionaries
    valid_memory = [m for m in memory if isinstance(m, dict)]
    
    # Define importance levels for nuanced sentiment analysis
    sentiment_importance = {
        "very positive": 3,
        "positive": 2,
        "slightly positive": 1,
        "neutral": 0,
        "slightly negative": -1,
        "negative": -2,
        "very negative": -3
    }
    
    # Categories and keywords that should be given higher priority
    important_categories = ["health", "finance", "personal information", "security", "career", "education"]
    important_keywords = ["important", "urgent", "critical", "must remember", "don't forget", "note this"]

    def memory_priority_score(memory_item):

        score = 0
        
        # Sentiment-based scoring
        score += sentiment_importance.get(memory_item.get("sentiment", "neutral"), 0)
        
        # Category-based scoring
        categories = memory_item.get("category", [])
        if any(category in important_categories for category in categories):
            score += 2
        
        # Keyword-based scoring
        if any(keyword in memory_item.get("user_input", "").lower() for keyword in important_keywords):
            score += 2
        
        # Recency could be considered implicitly by the order in the list, newer items come last
        return score
    
    # Prioritize memory by score and then by recency (most recent first)
    prioritized_memory = sorted(valid_memory, key=lambda x: (memory_priority_score(x), valid_memory.index(x)), reverse=True)
    
    # Truncate memory if exceeding limit
    if len(prioritized_memory) > limit:
        return prioritized_memory[:limit]
    
    return prioritized_memory


# Initialize OpenAI API with your key
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(prompt, personality_traits):
    personality_description = ", ".join([f"{trait}: {level}" for trait, level in personality_traits.items()])
    system_prompt = f"You are Nyra, a chatbot with the following personality traits: {personality_description}. Respond accordingly."
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].message.content

def chat_with_nyra():
    user_id = load_user_id()
    user_profile, user_id = load_user_profile(user_id)
    
    memory = {
        "short-term": load_memory(filename="nyra_memory_short_term.json"),
        "long-term": load_memory(filename="nyra_memory_long_term.json")
    }
    print(f"Nyra: Hi, {user_profile['name']}! What can I help with today?")
    
    while True:
        user_input = input(f"{user_profile['name']}: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print(f"Nyra: Goodbye, {user_profile['name']}! Have a great day!")
            break

        memory_context = recall_relevant_memory(memory["short-term"], user_input) \
                         or recall_relevant_memory(memory["long-term"], user_input)
        prompt = f"{memory_context} Current conversation: {user_input}"

        nyra_response = generate_response(prompt, user_profile["preferences"])
        print(f"Nyra: {nyra_response}")

        manage_memory(memory, {"user_input": user_input, "nyra_response": nyra_response})
        user_profile["history"].append({"input": user_input, "response": nyra_response})
        save_user_profile(user_id, user_profile)

# Start chatting with the enhanced Nyra
chat_with_nyra()
