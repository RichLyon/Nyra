#### Nyra Chatbot

Nyra is an intelligent chatbot designed to interact with users in a personalized manner, providing engaging conversations and maintaining memory across sessions. Nyra learns from each interaction, storing relevant information to enhance future conversations, and adapts its responses based on a dynamic personality profile. This chatbot is built using Python, OpenAI's GPT models, and various natural language processing (NLP) techniques.

## Features

### 1. **Personalized User Experience**
   - **Unique User Identification:** Each user is assigned a unique ID using UUID, which is stored in a `user_id.txt` file. This ID ensures that Nyra recognizes returning users and retrieves their specific profile and conversation history.
   - **User Profile Management:** Nyra stores user profiles in a `nyra_user_profiles.json` file, which includes the user's name, a history of interactions, and a set of personality preferences that influence Nyra's conversational style.
   - **Name-Based Interactions:** During the first interaction, Nyra asks for the user's name and uses it in all subsequent conversations, creating a more personalized and friendly experience.

### 2. **Dynamic Personality**
   - **Customizable Traits:** Nyra's personality traits (e.g., openness, humor, empathy) are customizable and stored within each user's profile. These traits are used to guide Nyra's responses, ensuring they align with the desired personality.
   - **Adaptive Responses:** Nyra adapts its conversational style based on the user's inputs and the stored personality traits, providing responses that are consistent with its defined character.

### 3. **Memory and Learning**
   - **Short-Term and Long-Term Memory:** Nyra maintains both short-term and long-term memories of conversations. These memories are stored separately in `nyra_memory_short_term.json` and `nyra_memory_long_term.json` files, allowing Nyra to recall relevant information in future interactions.
   - **Sentiment and Topic Analysis:** Nyra uses natural language processing (NLP) to analyze the sentiment and extract keywords from user inputs. This helps Nyra to categorize topics and respond appropriately.
   - **Prioritized Memory Management:** Nyra prioritizes memories based on their relevance and emotional impact, ensuring that important interactions are remembered while less significant ones are managed efficiently.

### 4. **Easy Setup and Configuration**
   - **Environment Configuration:** The app uses Python's `dotenv` to manage environment variables securely, including the OpenAI API key.
   - **Simple Interaction Loop:** Nyra engages users in a continuous conversation loop until the user decides to exit. This makes it easy to have ongoing interactions with Nyra without needing to restart the application.

## Getting Started

### Prerequisites
- **Python 3.7+**
- **OpenAI API Key:** You must have an OpenAI API key to use the GPT models. Set this up in a `.env` file.
- **Required Python Packages:** Install necessary packages using pip.

```bash
pip install openai textblob spacy python-dotenv
python -m spacy download en_core_web_sm
```

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/nyra-chatbot.git
   cd nyra-chatbot
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Environment:**
   - Create a `.env` file in the root directory and add your OpenAI API key:
   
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

4. **Run the Application:**

   ```bash
   python main.py
   ```

### Usage

When you run the application for the first time, Nyra will ask for your name and create a unique user profile. Your conversations will be stored and recalled in future interactions, allowing Nyra to interact with you in a more personalized and informed way.

### File Structure

- **`main.py`**: The core script that runs Nyra and handles user interactions.
- **`user_id.txt`**: Stores the unique user ID for the session.
- **`nyra_user_profiles.json`**: Stores user profiles including names, interaction history, and personality preferences.
- **`nyra_memory_short_term.json`**: Stores short-term memory interactions.
- **`nyra_memory_long_term.json`**: Stores long-term memory interactions.

### Customization

Nyra's personality and behavior can be customized by modifying the initial personality traits in the `main.py` script. You can adjust how Nyra responds by tweaking these settings according to your preferences.

### Contributing

Contributions to enhance Nyra's capabilities or improve its user experience are welcome. Feel free to fork the repository, make improvements, and submit a pull request.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Nyra is an evolving project designed to demonstrate the capabilities of modern conversational AI. Whether you're using it for learning, fun, or practical applications, Nyra is here to engage with you in meaningful and dynamic conversations.
