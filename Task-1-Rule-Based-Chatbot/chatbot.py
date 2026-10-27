"""
CodSoft Artificial Intelligence Internship - Task 1: Rule-Based Chatbot
======================================================================
Developer: Prit Panvalkar / CodSoft Intern
Python Compatibility: Python 3.8+ (Tested on Python 3.13.1 on Windows)

IMPORTANT ARCHITECTURAL NOTE:
------------------------------
This is a strictly RULE-BASED conversational agent. It does NOT use Large
Language Models (LLMs), neural networks, or generative AI APIs (such as OpenAI,
Gemini, or Claude).

How it works:
1. Input Normalization: Cleans incoming user text (lowercasing, trimming,
   whitespace collapsing, and punctuation stripping).
2. Intent Classification: Evaluates patterns using regular expressions and
   keyword matching in priority order.
3. Response Generation: Selects appropriate responses from curated, varied
   response pools using deterministic and pseudo-random selection.
4. Fallback Handling: Safely catches unrecognized inputs, empty inputs,
   and termination commands.
======================================================================
"""

import datetime
import random
import re
import sys
from typing import List, Optional, Tuple


class RuleBasedChatbot:
    """A lightweight, rule-based chatbot utilizing keyword & regex matching."""

    BOT_NAME = "CodSoft Assistant"

    EXIT_COMMANDS = {
        "exit",
        "quit",
        "bye",
        "goodbye",
        "close",
        "stop",
        "terminate",
        "cya",
        "see you",
    }

    def __init__(self, name: Optional[str] = None):
        """Initialize the chatbot with an optional custom name and response pools."""
        self.name = name or self.BOT_NAME

        # Greeting response pool
        self._greetings = [
            f"Hello! I am {self.name}. How can I assist you today?",
            f"Hi there! Welcome. How can I help you today?",
            f"Hey! Great to see you. What would you like to chat about?",
            f"Greetings! I am ready to help. Type 'help' if you need guidance.",
        ]

        # Well-being response pool
        self._well_being = [
            "I'm operating at peak efficiency! Thank you for asking. How are you doing today?",
            "Doing fantastic, thank you! Ready and eager to help you with your questions.",
            "All systems operational! How has your day been so far?",
            "I'm feeling great! How can I make your day easier?",
        ]

        # Gratitude response pool
        self._gratitude = [
            "You're very welcome! Let me know if there's anything else you need.",
            "Happy to help! Don't hesitate to ask more questions.",
            "Anytime! Glad I could be of assistance.",
            "My pleasure! Have a wonderful day ahead.",
        ]

        # Compliment response pool
        self._compliments = [
            "Thank you so much! That means a lot coming from you.",
            "You're very kind! I strive to be as helpful as possible.",
            "Appreciate the compliment! Teamwork makes the dream work.",
            "Thank you! You're pretty awesome yourself!",
        ]

        # Farewell response pool
        self._farewells = [
            f"Goodbye! It was a pleasure chatting with you. Have a great day!",
            f"Farewell! Take care and come back soon.",
            f"Bye! Feel free to return anytime you need help.",
            f"See you later! Keep learning and building amazing things.",
        ]

        # Jokes pool
        self._jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why did the Python programmer not wear glasses? Because they could see in C#!",
            "There are only 10 types of people in the world: those who understand binary, and those who don't.",
            "Why do Java developers wear glasses? Because they don't C#!",
            "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
            "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
        ]

        # Python & Programming study tips
        self._tips = [
            "Python Tip: Use list comprehensions for concise, readable transformations of data.",
            "Python Tip: Remember that Python dictionaries are insertion-ordered starting from Python 3.7!",
            "Study Tip: Practice deliberate coding by building small, focused projects rather than only watching tutorials.",
            "Code Quality Tip: Follow PEP 8 style guidelines to write readable, professional Python code.",
            "Debugging Tip: When encountering an error, read the full traceback from bottom to top.",
        ]

        # Unknown / Fallback response pool
        self._fallbacks = [
            "I'm not quite sure I understand that. Could you please rephrase or type 'help'?",
            "Interesting! As a rule-based chatbot, my vocabulary is bounded. Type 'help' to see what I can answer.",
            "I didn't catch that. Try asking about who I am, the time/date, study tips, a joke, or type 'help'.",
            "Sorry, that query doesn't match any of my predefined rules. Enter 'help' to view available commands.",
        ]

    # ------------------------------------------------------------------
    # Normalization & Utility Methods
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_input(raw_text: str) -> str:
        """
        Normalize raw text input by:
        1. Converting to lowercase.
        2. Stripping leading and trailing whitespace.
        3. Collapsing multiple internal whitespaces into a single space.
        4. Stripping trailing non-alphanumeric punctuation (!, ?, ., etc.)
           while preserving internal characters.
        """
        if not raw_text:
            return ""

        # Lowercase and whitespace strip
        cleaned = raw_text.lower().strip()

        # Collapse contiguous whitespaces
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Remove trailing punctuation like 'hello???' -> 'hello'
        cleaned = re.sub(r"[!?,.;:]+$", "", cleaned).strip()

        return cleaned

    def is_exit_command(self, user_input: str) -> bool:
        """Determine whether the user entered an exit/termination command."""
        normalized = self.normalize_input(user_input)
        return normalized in self.EXIT_COMMANDS

    def get_welcome_message(self) -> str:
        """Return the startup welcome banner and instructions."""
        border = "=" * 62
        return (
            f"\n{border}\n"
            f"  Welcome to {self.name}!\n"
            f"  (CodSoft Artificial Intelligence Internship - Task 1)\n"
            f"{border}\n"
            f"  * Architecture: Rule-Based Conversational Engine\n"
            f"  * Type 'help' at any time to see topics I can answer.\n"
            f"  * Type 'exit', 'quit', or 'bye' when you wish to leave.\n"
            f"{border}\n"
        )

    def get_help_message(self) -> str:
        """Return a formatted help guide outlining supported topics."""
        return (
            "Here are some topics and questions you can ask me:\n"
            "  - Greetings: 'hello', 'hi', 'hey', 'good morning'\n"
            "  - Identity: 'who are you?', 'what is your name?', 'introduce yourself'\n"
            "  - Creator: 'who made you?', 'who created you?'\n"
            "  - Well-being: 'how are you?', 'how is it going?'\n"
            "  - Date & Time: 'what time is it?', 'what is today's date?'\n"
            "  - Entertainment: 'tell me a joke', 'make me laugh'\n"
            "  - Study & Coding: 'give me a python tip', 'study advice'\n"
            "  - Weather: 'what is the weather like?'\n"
            "  - Appreciation: 'thank you', 'you are awesome'\n"
            "  - Exit: 'exit', 'quit', 'bye', 'goodbye'"
        )

    # ------------------------------------------------------------------
    # Core Pattern Matching & Response Logic
    # ------------------------------------------------------------------

    def get_response(self, user_input: str) -> str:
        """
        Process the user's input against predefined rules and return a response.

        Evaluates rules sequentially using regular expressions and keyword
        membership to ensure predictable, deterministic behavior.
        """
        normalized = self.normalize_input(user_input)

        # 1. Handle empty or whitespace-only input
        if not normalized:
            return (
                "It looks like you didn't say anything! "
                "Feel free to ask a question or type 'help' to see what I can do."
            )

        # 2. Check for exit commands
        if self.is_exit_command(normalized):
            return random.choice(self._farewells)

        # 3. Help and capabilities
        if normalized in {"help", "commands", "options"} or re.search(
            r"\b(what can you do|what are your features|how do you work|help me|commands?|options?)\b",
            normalized,
        ):
            return self.get_help_message()

        # 4. Identity & Name questions
        if re.search(
            r"\b(what is your name|who are you|what should i call you|what('s|s) your name|introduce yourself)\b",
            normalized,
        ):
            return (
                f"I am {self.name}, an intelligent rule-based chatbot developed for "
                f"the CodSoft Artificial Intelligence Internship. I process your text "
                f"using keyword analysis and regular expressions!"
            )

        # 5. Creator / Origin questions
        if re.search(
            r"\b(who (created|made|built|developed|programmed) you|who is your (creator|author|maker))\b",
            normalized,
        ) or "codsoft" in normalized:
            return (
                f"I was built by a CodSoft AI Intern using Python 3.13! "
                f"My logic is completely modular and rule-driven, without external ML dependencies."
            )

        # 6. Greetings
        if re.search(
            r"^(hi|hello|hey|heya|howdy|sup|greetings|hola)\b", normalized
        ) or re.search(
            r"\b(good (morning|afternoon|evening|day))\b", normalized
        ):
            return random.choice(self._greetings)

        # 7. Well-being inquiries
        if re.search(
            r"\b(how are you|how('s|s) it going|how do you do|how are things|how are you doing)\b",
            normalized,
        ):
            return random.choice(self._well_being)

        # 8. Date and Time inquiries (dynamic lookup using Python's datetime)
        if re.search(r"\b(what time is it|current time|tell me the time)\b", normalized):
            now = datetime.datetime.now().strftime("%I:%M %p")
            return f"The current local time is {now}."

        if re.search(
            r"\b(what is the date|today('s|s) date|what date is it|what is today)\b",
            normalized,
        ):
            today = datetime.datetime.now().strftime("%A, %B %d, %Y")
            return f"Today is {today}."

        if re.search(r"\b(time|date)\b", normalized):
            now_dt = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
            return f"Right now it is {now_dt}."

        # 9. Jokes & Entertainment
        if re.search(r"\b(jokes?|make me laugh|tell me a joke|funny)\b", normalized):
            return random.choice(self._jokes)

        # 10. Study & Python programming tips
        if re.search(
            r"\b(python tips?|coding tips?|study tips?|programming tips?|advice|tips?)\b",
            normalized,
        ):
            return random.choice(self._tips)

        # 11. Weather inquiries (explains rule-based limitation clearly)
        if re.search(r"\b(weather|forecast|temperature|is it raining)\b", normalized):
            return (
                "Since I am a lightweight rule-based chatbot running offline without live API access, "
                "I cannot check real-time weather sensors. Please peek out of your window or consult a weather app!"
            )

        # 12. Gratitude & Courtesy
        if re.search(r"\b(thank you|thanks|thx|appreciate it|much obliged)\b", normalized):
            return random.choice(self._gratitude)

        # 13. Compliments & Affirmations
        if re.search(
            r"\b(you are (awesome|cool|great|smart|clever|amazing|helpful|the best)|good job|well done)\b",
            normalized,
        ):
            return random.choice(self._compliments)

        # 14. Age / Sentience inquiries
        if re.search(r"\b(how old are you|what is your age|are you (human|alive|sentient|real))\b", normalized):
            return (
                f"I don't have a biological age or physical body! I am a rule-based computer program "
                f"running on Python {sys.version.split()[0]}."
            )

        # 15. Fallback for unknown / unhandled input
        return random.choice(self._fallbacks)

    # ------------------------------------------------------------------
    # Interactive CLI Chat Loop
    # ------------------------------------------------------------------

    def start_chat(self) -> None:
        """Run the interactive conversational CLI loop."""
        print(self.get_welcome_message())

        while True:
            try:
                raw_input_text = input("You: ")
            except (KeyboardInterrupt, EOFError):
                print("\n\nSession interrupted. Exiting...")
                print(f"{self.name}: {random.choice(self._farewells)}")
                break

            # Check for exit before generating response
            if self.is_exit_command(raw_input_text):
                farewell = self.get_response(raw_input_text)
                print(f"{self.name}: {farewell}")
                break

            response = self.get_response(raw_input_text)
            print(f"{self.name}: {response}\n")


def main() -> None:
    """Entry point when running chatbot directly."""
    bot = RuleBasedChatbot()
    bot.start_chat()


if __name__ == "__main__":
    main()
