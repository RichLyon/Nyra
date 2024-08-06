import openai
from dotenv import load_dotenv
import os
import json
from textblob import TextBlob
import spacy

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

# Example: Increase empathy by 1 level
# updated_personality = update_personality(current_personality, "empathy", 1)
# save_personality(updated_personality)
# print(updated_personality)

# Memory file
memory_file = "nyra_memory.json"

# Function to load memory
def load_memory(filename=memory_file):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Return a dictionary with 'short-term','long-term' keys, each initialized to an empty list
        return {"short-term": [], "long-term": [], "sentiment": [], "keywords": [], "category": []}

# Function to save memory
def save_memory(memory, filename=memory_file):
    with open(filename, "w") as f:
        json.dump(memory, f)

# Function to add a new memory
def add_memory(memory, interaction, memory_type="short-term"):
    """
    Add a new memory to Nyra's memory bank.
    
    :param memory: The current memory dictionary with 'short-term' and 'long-term' keys.
    :param interaction: A dictionary containing user input, Nyra's response, and metadata.
    :param memory_type: Which memory segment to add to ('short-term' or 'long-term').
    """
    if memory_type in memory and isinstance(memory[memory_type], list):
        memory[memory_type].append(interaction)
    else:
        print("Error: Invalid memory type or structure")
    save_memory(memory)


def manage_memory(memory, interaction):
    print("Manage_memory called")
    """
    Add the interaction to short-term or long-term memory based on its importance.
    
    :param memory: The current memory structure with both short-term and long-term memory.
    :param interaction: The interaction to be added to memory.
    """
    if should_store_long_term(interaction["user_input"]):
        add_memory(memory, interaction, memory_type="long-term")
    else:
        add_memory(memory, interaction, memory_type="short-term")
    
    # Prioritize and save memories
    memory["short-term"] = prioritize_memory(memory["short-term"])
    save_memory(memory["short-term"], filename="nyra_memory_short_term.json")
    save_memory(memory["long-term"], filename="nyra_memory_long_term.json")

def extract_keywords(text):
    # Simple keyword extraction, can be expanded with NLP libraries like spaCy
    return text.lower().split()

def categorize_topics(text):
    # Simple topic categorization, can be expanded with more complex models (spaCy here)
    if "space" in text.lower():
        return ["astronomy", "science"]
    elif "emotion" in text.lower():
        return ["psychology", "well-being"]
    return ["general"]

def analyze_sentiment(text):
    # Placeholder for sentiment analysis
    if any(word in text.lower() for word in ["happy", "joy", "great"]):
        return "positive"
    elif any(word in text.lower() for word in ["sad", "angry", "bad"]):
        return "negative"
    return "neutral"

# Example usage with enhanced memory
current_memory = load_memory()
new_interaction = {
    "user_input": "I'm feeling great about space exploration!",
    "nyra_response": "That's awesome! Space exploration is full of wonders.",
    "sentiment": analyze_sentiment(current_memory),
    "keywords": extract_keywords(current_memory),
    "category": categorize_topics(current_memory)
}
add_memory(current_memory, new_interaction)
print(current_memory)

def recall_relevant_memory(memory_segment, current_input):
    """
    Recall the most relevant memory from the given memory segment based on the current user input.
    
    :param memory_segment: The segment of memory to search (either short-term or long-term).
    :param current_input: The current input from the user.
    :return: A string summarizing relevant past interactions or an empty string if none are found.
    """
    relevant_memories = []
    keywords = extract_keywords(current_input)
    
    for interaction in memory_segment:
        if isinstance(interaction, dict):  # Check if interaction is a dictionary
            interaction_keywords = interaction.get("keywords", [])
            if any(keyword in interaction_keywords for keyword in keywords):
                relevant_memories.append(interaction)
        else:
            print(f"Warning: Encountered a non-dictionary interaction: {interaction}")
    
    if relevant_memories:
        return "In previous conversations, you mentioned: " + \
               "; ".join([f"{m['user_input']} ({m['sentiment']})" for m in relevant_memories])
    
    return ""

# Example usage
current_memory = load_memory()
memory_context = recall_relevant_memory(current_memory["short-term"], "space exploration")
print(memory_context)

add_memory(current_memory, new_interaction, memory_type="short-term")
print(current_memory)

# Testing the recall function

memory_context = recall_relevant_memory(current_memory["short-term"], "space exploration")
print(memory_context)

# Function to decide if an interaction should be stored long-term
def should_store_long_term(user_input):
    # Placeholder logic: store long-term if input mentions specific topics or is highly emotional
    if "important" in user_input.lower() or analyze_sentiment(user_input) in ["positive", "negative"]:
        return True
    return False

def prioritize_memory(memory, limit=100):
    print("prioritize_memory called")
    """
    Prioritize memory by importance and relevance, discarding the least important if memory exceeds a limit.
    
    :param memory: The current memory list.
    :param limit: Maximum number of memories to retain.
    :return: Optimized memory list.
    """
    # Ensure all items in memory are dictionaries
    valid_memory = [m for m in memory if isinstance(m, dict)]
    
    # Simple prioritization: keep recent and high-sentiment memories
    prioritized_memory = sorted(valid_memory, key=lambda x: x.get("sentiment", "neutral"), reverse=True)
    
    # Truncate if exceeding limit
    if len(prioritized_memory) > limit:
        return prioritized_memory[:limit]
    
    return prioritized_memory

# Example usage of prioritizing memory
current_memory = load_memory()
current_memory["short-term"] = prioritize_memory(current_memory["short-term"])


def analyze_sentiment(text):
    """
    Analyze sentiment using TextBlob.
    
    :param text: The text to analyze.
    :return: Sentiment category ('positive', 'neutral', 'negative').
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

nlp = spacy.load("en_core_web_sm")

def categorize_topics(text):
    """
    Categorize topics using spaCy for Named Entity Recognition (NER).
    
    :param text: The text to analyze.
    :return: List of identified topics.
    """
    doc = nlp(text)
    topics = []
    
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "PERSON", "NORP", "FAC", "LOC", "PRODUCT"]:
            topics.append(ent.text)
    
    return topics if topics else ["general"]

# Initialize OpenAI API with your key
load_dotenv()

def load_user_profile(user_id, filename="nyra_user_profiles.json"):
    try:
        with open(filename, "r") as f:
            profiles = json.load(f)
            return profiles.get(user_id, {"preferences": {}, "history": []})
    except FileNotFoundError:
        return {"preferences": {}, "history": []}

def save_user_profile(user_id, profile, filename="nyra_user_profiles.json"):
    profiles = load_user_profile(None, filename)
    profiles[user_id] = profile
    with open(filename, "w") as f:
        json.dump(profiles, f)

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(prompt, personality_traits):
    """
    Generate a response from Nyra based on the given prompt and personality traits.
    
    :param prompt: The user's input prompt.
    :param personality_traits: A dictionary of Nyra's personality traits.
    :return: The response generated by GPT-4o mini.
    """
    # Create a system prompt that incorporates personality dimensions
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


def chat_with_nyra(user_id="default_user"):
    # Load user profile and memory segments
    user_profile = load_user_profile(user_id)
    memory = {
        "short-term": load_memory(filename="nyra_memory_short_term.json"),
        "long-term": load_memory(filename="nyra_memory_long_term.json")
    }
    
    print("Nyra: Hi, I'm Nyra! How can I assist you today?")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Nyra: Goodbye! Have a great day!")
            break
        
        # Recall relevant memories, prioritizing short-term over long-term
        memory_context = recall_relevant_memory(memory["short-term"], user_input) \
                         or recall_relevant_memory(memory["long-term"], user_input)
        
        # Generate response using the personality traits and memory context
        prompt = f"{memory_context} Current conversation: {user_input}"
        nyra_response = generate_response(prompt, user_profile["preferences"])
        
        print(f"Nyra: {nyra_response}")
        
        # Add the interaction to the appropriate memory segment
        manage_memory(memory, {"user_input": user_input, "nyra_response": nyra_response})
        
        # Optionally update user profile based on the conversation
        user_profile["history"].append({"input": user_input, "response": nyra_response})
        save_user_profile(user_id, user_profile)


# Start chatting with the enhanced Nyra
chat_with_nyra()