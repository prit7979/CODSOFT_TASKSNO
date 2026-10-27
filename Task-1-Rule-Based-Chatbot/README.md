# CodSoft Artificial Intelligence Internship - Task 1: Rule-Based Chatbot

A lightweight, beginner-friendly, and deterministic rule-based chatbot implemented in Python. Designed and tested on **Python 3.13.1** on Windows, this project demonstrates natural language processing concepts using rule-driven pattern matching and keyword analysis without external dependencies.

---

## 📌 Architectural Overview & Rule-Based Design

> **Important Note:** This chatbot is purely **rule-based** and does **not** rely on Large Language Models (LLMs), neural networks, or paid external APIs (such as OpenAI, Gemini, or Claude).

The conversational flow operates on a deterministic pipeline:
1. **Input Normalization:** Strips leading/trailing whitespace, converts all tokens to lowercase, collapses consecutive spaces, and removes terminal punctuation marks (`!`, `?`, `.`) to improve matching accuracy.
2. **Intent Matching:** Sequentially evaluates user input using prioritized regular expressions (`re`) and keyword collections.
3. **Response Selection:** Pulls from curated response pools with pseudo-random selection (`random.choice`) to provide dynamic and natural conversational variety.
4. **Fallback & Safety Guards:** Safely handles empty inputs, whitespace-only entries, keyboard interruptions (`Ctrl+C`, `EOFError`), and unrecognized queries with informative guidance.

---

## ✨ Features

- **Warm Welcome & Clear Guidance:** Welcoming CLI banner displaying instructions and the `help` command.
- **Robust Input Cleaning:** Case-insensitive matching, whitespace tolerance, and trailing punctuation stripping.
- **Rich Predefined Rule Categories:**
  - **Greetings:** `hello`, `hi`, `hey`, `good morning`, `howdy`, `sup`, `greetings`.
  - **Identity & Name Inquiries:** `who are you?`, `what is your name?`, `introduce yourself`.
  - **Creator & Internship Context:** `who made you?`, `who created you?`, `tell me about CodSoft`.
  - **Help & Capabilities:** `help`, `commands`, `options`, `what can you do?`.
  - **Date & Time Queries:** Real-time dynamic local date and time via Python's standard `datetime` module.
  - **Well-Being Questions:** `how are you?`, `how's it going?`, `how do you do?`.
  - **Humor & Entertainment:** Curated developer and computer science jokes (`tell me a joke`, `make me laugh`).
  - **Programming & Study Tips:** Python best practices and studying advice (`python tip`, `coding tip`).
  - **Polite Exchanges & Gratitude:** `thank you`, `thanks`, `appreciate it`.
  - **Compliments & Encouragement:** `you are awesome`, `you are smart`, `good job`.
  - **Scope & Limitations Explanation:** Inquiries regarding real-time weather sensors politely clarify offline rule-based bounds.
  - **Graceful Termination:** `exit`, `quit`, `bye`, `goodbye`, `cya`, `stop`.
- **Varied Responses:** Employs response pools so greetings, jokes, pleasantries, and fallbacks don't sound repetitive.
- **Zero External Dependencies:** Built entirely with Python's standard library (`re`, `random`, `datetime`, `sys`, `unittest`).
- **Comprehensive Unit Testing:** Automated tests covering normalization, rules, exit triggers, and mock user sessions.

---

## 📁 Project Directory Structure

```text
Task-1-Rule-Based-Chatbot/
├── .venv/               # Isolated Python virtual environment (ignored by git)
├── chatbot.py           # Main chatbot implementation & interactive CLI loop
├── test_chatbot.py      # Automated unit tests using Python's unittest framework
├── requirements.txt     # Dependency specification (zero external libraries)
└── README.md            # Comprehensive project documentation
```

---

## 🚀 Setup & Execution (Windows)

All commands below are tailored for Windows using PowerShell or Command Prompt.

### 1. Navigate to the Project Directory

```powershell
cd D:\CODSOFT_TASKSNO\Task-1-Rule-Based-Chatbot
```

### 2. Set Up Virtual Environment

If the virtual environment is not already created:

```powershell
py -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\activate
```

*(You will see `(.venv)` in your command prompt indicating the environment is active.)*

### 3. Verify Dependencies

Because this project uses the standard library exclusively, no external packages are required:

```powershell
pip install -r requirements.txt
```

---

## 💬 Running the Chatbot

With the virtual environment activated:

```powershell
python chatbot.py
```

Or run directly without activating the virtual environment:

```powershell
.\.venv\Scripts\python.exe chatbot.py
```

*(Note: If Windows prompts that `python was not found` due to the Windows Store App execution alias, always activate the venv first or invoke `.\.venv\Scripts\python.exe` / `py chatbot.py` directly).*

### Sample Interaction

```text
==============================================================
  Welcome to CodSoft Assistant!
  (CodSoft Artificial Intelligence Internship - Task 1)
==============================================================
  * Architecture: Rule-Based Conversational Engine
  * Type 'help' at any time to see topics I can answer.
  * Type 'exit', 'quit', or 'bye' when you wish to leave.
==============================================================

You: Hello!
CodSoft Assistant: Hi there! Welcome. How can I help you today?

You: What is your name?
CodSoft Assistant: I am CodSoft Assistant, an intelligent rule-based chatbot developed for the CodSoft Artificial Intelligence Internship. I process your text using keyword analysis and regular expressions!

You: What time is it?
CodSoft Assistant: The current local time is 06:30 PM.

You: Tell me a joke
CodSoft Assistant: Why do programmers prefer dark mode? Because light attracts bugs!

You: Give me a python tip
CodSoft Assistant: Python Tip: Use list comprehensions for concise, readable transformations of data.

You: Thank you!
CodSoft Assistant: You're very welcome! Let me know if there's anything else you need.

You: Bye
CodSoft Assistant: Goodbye! It was a pleasure chatting with you. Have a great day!
```

---

## 🧪 Running Unit Tests

Automated unit tests are provided in `test_chatbot.py` to verify:
- Input normalization (lowercasing, trimming, collapsing internal whitespace, punctuation removal).
- Keyword and regex pattern matching across all supported topics.
- Empty and unrecognized inputs fallback handling.
- Exit command detection.
- Simulated CLI sessions using `unittest.mock`.

Run the test suite with:

```powershell
.\.venv\Scripts\python -m unittest test_chatbot.py -v
```

or via the `py` launcher:

```powershell
py -m unittest test_chatbot.py -v
```

---

## ⚠️ Limitations

1. **Vocabulary & Keyword Bounds:** The chatbot only responds intelligently to inputs matching its predefined keywords and regex patterns. Out-of-domain queries trigger a polite fallback message.
2. **Context & State Retention:** It operates primarily in a single-turn request-response pattern and does not retain multi-turn conversational history.
3. **No Real-Time Internet Data:** The bot runs locally and offline; it cannot fetch live weather, stocks, or web articles.
4. **Not an LLM:** Unlike generative language models, it does not invent new sentences or generate hallucinated facts; all outputs originate from defined rule logic.
