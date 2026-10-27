"""
Unit Tests for Rule-Based Chatbot
=================================
CodSoft Artificial Intelligence Internship - Task 1
Uses Python's built-in unittest framework (zero external dependencies).
"""

import io
import sys
import unittest
from unittest.mock import patch

from chatbot import RuleBasedChatbot


class TestRuleBasedChatbot(unittest.TestCase):
    """Comprehensive test suite verifying chatbot rules, normalization, and CLI flow."""

    def setUp(self):
        """Instantiate a fresh chatbot instance for each test."""
        self.bot = RuleBasedChatbot()

    # ------------------------------------------------------------------
    # 1. Normalization Tests
    # ------------------------------------------------------------------

    def test_normalize_empty_and_whitespace(self):
        """Test normalization of empty strings and whitespace."""
        self.assertEqual(RuleBasedChatbot.normalize_input(""), "")
        self.assertEqual(RuleBasedChatbot.normalize_input("   "), "")
        self.assertEqual(RuleBasedChatbot.normalize_input("\t\n"), "")

    def test_normalize_lowercasing(self):
        """Test that uppercase characters are properly lowercased."""
        self.assertEqual(
            RuleBasedChatbot.normalize_input("HELLO WORLD"), "hello world"
        )
        self.assertEqual(
            RuleBasedChatbot.normalize_input("WhAt Is YoUr NaMe"),
            "what is your name",
        )

    def test_normalize_whitespace_cleanup(self):
        """Test collapsing multiple spaces, tabs, and trims."""
        self.assertEqual(
            RuleBasedChatbot.normalize_input("   hello    there   friend   "),
            "hello there friend",
        )

    def test_normalize_trailing_punctuation(self):
        """Test stripping of trailing punctuation marks while preserving words."""
        self.assertEqual(RuleBasedChatbot.normalize_input("hello!"), "hello")
        self.assertEqual(RuleBasedChatbot.normalize_input("how are you???"), "how are you")
        self.assertEqual(RuleBasedChatbot.normalize_input("good morning..."), "good morning")
        self.assertEqual(RuleBasedChatbot.normalize_input("thanks!!?!"), "thanks")

    # ------------------------------------------------------------------
    # 2. Exit Command Tests
    # ------------------------------------------------------------------

    def test_is_exit_command_positive(self):
        """Test that standard exit commands are recognized across cases and whitespace."""
        for cmd in ["exit", "EXIT", "  quit  ", "bye", "goodbye", "cya", "stop"]:
            self.assertTrue(
                self.bot.is_exit_command(cmd),
                f"Failed to identify '{cmd}' as an exit command",
            )

    def test_is_exit_command_negative(self):
        """Test that non-exit commands are not flagged as exit."""
        for phrase in ["hello", "tell me a joke", "how are you", "excited", "quite good"]:
            self.assertFalse(
                self.bot.is_exit_command(phrase),
                f"Incorrectly identified '{phrase}' as an exit command",
            )

    # ------------------------------------------------------------------
    # 3. Intent & Response Matching Tests
    # ------------------------------------------------------------------

    def test_empty_input_response(self):
        """Test response when empty or whitespace-only inputs are provided."""
        empty_responses = [
            self.bot.get_response(""),
            self.bot.get_response("   "),
            self.bot.get_response("\t\n"),
        ]
        for resp in empty_responses:
            self.assertIn("didn't say anything", resp.lower())

    def test_greeting_responses(self):
        """Test response to various greeting patterns."""
        sample_greetings = ["hello", "HI THERE", "  Hey!  ", "good morning", "howdy"]
        for greeting in sample_greetings:
            response = self.bot.get_response(greeting)
            self.assertIn(response, self.bot._greetings)

    def test_identity_and_name_responses(self):
        """Test response to questions about chatbot's identity and name."""
        identity_queries = [
            "who are you?",
            "what is your name?",
            "what's your name",
            "introduce yourself",
        ]
        for query in identity_queries:
            response = self.bot.get_response(query)
            self.assertIn(self.bot.name, response)
            self.assertIn("rule-based", response.lower())

    def test_creator_and_codsoft_responses(self):
        """Test inquiries about the bot's creator or CodSoft."""
        creator_queries = [
            "who created you?",
            "who made you",
            "who is your author",
            "tell me about codsoft",
        ]
        for query in creator_queries:
            response = self.bot.get_response(query)
            self.assertIn("codsoft", response.lower())

    def test_help_and_options(self):
        """Test that typing 'help' or asking for features returns the help guide."""
        help_queries = ["help", "commands", "options", "what can you do?"]
        for query in help_queries:
            response = self.bot.get_response(query)
            self.assertIn("topics and questions", response.lower())
            self.assertIn("greetings", response.lower())

    def test_well_being_responses(self):
        """Test responses to 'how are you' style questions."""
        well_being_queries = [
            "how are you?",
            "how are you doing",
            "how's it going?",
            "how do you do",
        ]
        for query in well_being_queries:
            response = self.bot.get_response(query)
            self.assertIn(response, self.bot._well_being)

    def test_date_and_time_responses(self):
        """Test dynamic time and date generation."""
        time_resp = self.bot.get_response("what time is it?")
        self.assertIn("current local time is", time_resp.lower())

        date_resp = self.bot.get_response("what is today's date?")
        self.assertIn("today is", date_resp.lower())

        general_resp = self.bot.get_response("tell me the date and time")
        self.assertIn("right now it is", general_resp.lower())

    def test_jokes_responses(self):
        """Test that asking for a joke returns an item from the jokes pool."""
        joke_queries = ["tell me a joke", "make me laugh", "got any jokes?"]
        for query in joke_queries:
            response = self.bot.get_response(query)
            self.assertIn(response, self.bot._jokes)

    def test_tips_responses(self):
        """Test tips and study advice queries."""
        tip_queries = ["python tip", "coding tip", "give me study advice"]
        for query in tip_queries:
            response = self.bot.get_response(query)
            self.assertIn(response, self.bot._tips)

    def test_weather_limitation_explanation(self):
        """Test that weather questions clarify rule-based offline limitations."""
        weather_resp = self.bot.get_response("what is the weather like?")
        self.assertIn("offline", weather_resp.lower())

    def test_gratitude_responses(self):
        """Test responses to thanks and appreciation."""
        gratitude_queries = ["thank you!", "thanks a lot", "appreciate it"]
        for query in gratitude_queries:
            response = self.bot.get_response(query)
            self.assertIn(response, self.bot._gratitude)

    def test_compliments_responses(self):
        """Test responses to compliments."""
        compliments = ["you are awesome", "you are smart!", "good job"]
        for comp in compliments:
            response = self.bot.get_response(comp)
            self.assertIn(response, self.bot._compliments)

    def test_age_and_sentience(self):
        """Test questions about age and sentience."""
        age_queries = ["how old are you?", "are you human?"]
        for query in age_queries:
            response = self.bot.get_response(query)
            self.assertIn("rule-based", response.lower())

    def test_unknown_fallback_responses(self):
        """Test that arbitrary unrecognized strings fall back cleanly."""
        unrecognized_queries = [
            "qwertyuiop asdfghjkl",
            "supercalifragilisticexpialidocious",
            "what is the meaning of quantum entanglement in deep space?",
        ]
        for query in unrecognized_queries:
            response = self.bot.get_response(query)
            self.assertIn(response, self.bot._fallbacks)

    def test_farewell_response_on_exit(self):
        """Test that exit commands return a response from the farewell pool."""
        for exit_cmd in ["bye", "goodbye", "quit", "exit"]:
            response = self.bot.get_response(exit_cmd)
            self.assertIn(response, self.bot._farewells)

    # ------------------------------------------------------------------
    # 4. Interactive CLI Loop Simulation Test
    # ------------------------------------------------------------------

    def test_start_chat_loop_with_exit(self):
        """Simulate an interactive user session that ends with an exit command."""
        user_inputs = ["hello", "who are you?", "tell me a joke", "exit"]

        with patch("builtins.input", side_effect=user_inputs):
            with patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
                self.bot.start_chat()
                output = fake_stdout.getvalue()

                # Verify welcome banner was printed
                self.assertIn("Welcome to CodSoft Assistant", output)
                # Verify identity response was printed
                self.assertIn("intelligent rule-based chatbot", output)
                # Verify loop terminated gracefully
                self.assertIn("CodSoft Assistant:", output)

    def test_start_chat_loop_keyboard_interrupt(self):
        """Simulate a user pressing Ctrl+C (KeyboardInterrupt)."""
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
                self.bot.start_chat()
                output = fake_stdout.getvalue()
                self.assertIn("Session interrupted. Exiting...", output)


if __name__ == "__main__":
    unittest.main()
