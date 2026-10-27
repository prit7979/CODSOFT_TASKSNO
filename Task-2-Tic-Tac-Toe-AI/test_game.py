"""
Unit Tests for Tic-Tac-Toe Game and Minimax AI
==============================================
CodSoft Artificial Intelligence Internship - Task 2
Uses Python's built-in unittest framework (zero external dependencies).
"""

import random
import unittest
from typing import List

from ai import get_best_move, minimax, order_moves
from game import TicTacToe


class TestTicTacToeBoard(unittest.TestCase):
    """Test suite for the TicTacToe board state and rules engine."""

    def setUp(self):
        """Create a fresh TicTacToe board before each test."""
        self.game = TicTacToe()

    def test_initial_board_state(self):
        """Verify the board initializes empty with 9 available moves."""
        self.assertEqual(len(self.game.board), 9)
        self.assertTrue(all(cell == TicTacToe.EMPTY for cell in self.game.board))
        self.assertEqual(self.game.available_moves(), list(range(9)))
        self.assertFalse(self.game.is_board_full())
        self.assertFalse(self.game.is_draw())
        self.assertFalse(self.game.is_game_over())
        self.assertIsNone(self.game.check_winner())

    def test_valid_moves(self):
        """Verify valid moves update the board correctly."""
        self.assertTrue(self.game.make_move(0, TicTacToe.PLAYER_X))
        self.assertEqual(self.game.board[0], TicTacToe.PLAYER_X)
        self.assertNotIn(0, self.game.available_moves())
        self.assertEqual(len(self.game.available_moves()), 8)

    def test_invalid_and_out_of_bounds_moves(self):
        """Verify out-of-bounds indices are rejected."""
        self.assertFalse(self.game.is_valid_move(-1))
        self.assertFalse(self.game.is_valid_move(9))
        self.assertFalse(self.game.is_valid_move(100))
        self.assertFalse(self.game.make_move(-1, TicTacToe.PLAYER_X))
        self.assertFalse(self.game.make_move(9, TicTacToe.PLAYER_O))

    def test_occupied_cell_prevention(self):
        """Verify an occupied cell cannot be overwritten by another move."""
        self.assertTrue(self.game.make_move(4, TicTacToe.PLAYER_X))
        # Attempt to overwrite cell 4 with 'O'
        self.assertFalse(self.game.is_valid_move(4))
        self.assertFalse(self.game.make_move(4, TicTacToe.PLAYER_O))
        self.assertEqual(self.game.board[4], TicTacToe.PLAYER_X)

    def test_undo_move(self):
        """Verify undo_move restores the cell to empty."""
        self.game.make_move(4, TicTacToe.PLAYER_X)
        self.assertEqual(self.game.board[4], TicTacToe.PLAYER_X)
        self.game.undo_move(4)
        self.assertEqual(self.game.board[4], TicTacToe.EMPTY)
        self.assertIn(4, self.game.available_moves())

    def test_clone_integrity(self):
        """Verify clone creates an independent deep copy of the board."""
        self.game.make_move(0, TicTacToe.PLAYER_X)
        clone = self.game.clone()
        clone.make_move(1, TicTacToe.PLAYER_O)

        self.assertEqual(self.game.board[0], TicTacToe.PLAYER_X)
        self.assertEqual(self.game.board[1], TicTacToe.EMPTY)  # Original untouched
        self.assertEqual(clone.board[1], TicTacToe.PLAYER_O)

    def test_horizontal_wins(self):
        """Test win detection across all three rows."""
        # Row 1 (0, 1, 2)
        self.game.board = ["X", "X", "X", " ", " ", " ", " ", " ", " "]
        self.assertEqual(self.game.check_winner(), "X")
        self.assertTrue(self.game.is_game_over())

        # Row 2 (3, 4, 5)
        self.game.board = [" ", " ", " ", "O", "O", "O", " ", " ", " "]
        self.assertEqual(self.game.check_winner(), "O")

        # Row 3 (6, 7, 8)
        self.game.board = [" ", " ", " ", " ", " ", " ", "X", "X", "X"]
        self.assertEqual(self.game.check_winner(), "X")

    def test_vertical_wins(self):
        """Test win detection across all three columns."""
        # Column 1 (0, 3, 6)
        self.game.board = ["O", " ", " ", "O", " ", " ", "O", " ", " "]
        self.assertEqual(self.game.check_winner(), "O")

        # Column 2 (1, 4, 7)
        self.game.board = [" ", "X", " ", " ", "X", " ", " ", "X", " "]
        self.assertEqual(self.game.check_winner(), "X")

        # Column 3 (2, 5, 8)
        self.game.board = [" ", " ", "O", " ", " ", "O", " ", " ", "O"]
        self.assertEqual(self.game.check_winner(), "O")

    def test_diagonal_wins(self):
        """Test win detection across both diagonals."""
        # Main diagonal (0, 4, 8)
        self.game.board = ["X", " ", " ", " ", "X", " ", " ", " ", "X"]
        self.assertEqual(self.game.check_winner(), "X")

        # Anti diagonal (2, 4, 6)
        self.game.board = [" ", " ", "O", " ", "O", " ", "O", " ", " "]
        self.assertEqual(self.game.check_winner(), "O")

    def test_draw_detection(self):
        """Test detection of full-board draw state."""
        # Cat's game (draw)
        # X O X
        # X O O
        # O X X
        self.game.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
        self.assertTrue(self.game.is_board_full())
        self.assertIsNone(self.game.check_winner())
        self.assertTrue(self.game.is_draw())
        self.assertTrue(self.game.is_game_over())

    def test_not_draw_when_cells_remain(self):
        """Verify unfinished games are not counted as draws."""
        self.game.board = ["X", "O", "X", "X", " ", "O", "O", "X", "X"]
        self.assertFalse(self.game.is_draw())
        self.assertFalse(self.game.is_board_full())

    def test_board_display_formatting(self):
        """Verify the board display string contains grid rows and numbering."""
        display = self.game.get_board_display()
        self.assertIn("1 | 2 | 3", display)
        self.assertIn("4 | 5 | 6", display)
        self.assertIn("7 | 8 | 9", display)
        self.assertIn("---+---+---", display)


class TestMinimaxAI(unittest.TestCase):
    """Test suite verifying Minimax AI optimality, tactics, and invincibility."""

    def setUp(self):
        self.game = TicTacToe()

    def test_ai_takes_immediate_winning_move_row(self):
        """AI must take the immediate winning cell to win the game."""
        # Board:
        # O O .
        # X X .
        # . . .
        # AI is 'O'. Cell 2 delivers an immediate win!
        self.game.board = ["O", "O", " ", "X", "X", " ", " ", " ", " "]
        best = get_best_move(self.game, ai_player="O")
        self.assertEqual(best, 2)

    def test_ai_takes_immediate_winning_move_diagonal(self):
        """AI must complete a winning diagonal when available."""
        # Board:
        # X . .
        # . X .
        # . . .
        # AI is 'X'. Cell 8 completes diagonal win.
        self.game.board = ["X", "O", " ", "O", "X", " ", " ", " ", " "]
        best = get_best_move(self.game, ai_player="X")
        self.assertEqual(best, 8)

    def test_ai_blocks_immediate_opponent_win(self):
        """AI must block the opponent's immediate winning move."""
        # Board:
        # X X .
        # O . .
        # . . .
        # Human is 'X', threat at index 2. AI is 'O', must choose 2!
        self.game.board = ["X", "X", " ", "O", " ", " ", " ", " ", " "]
        best = get_best_move(self.game, ai_player="O")
        self.assertEqual(best, 2)

    def test_ai_blocks_immediate_opponent_column_win(self):
        """AI must block an opponent's vertical column threat."""
        # Board:
        # X . .
        # X O .
        # . . .
        # Human is 'X' at 0 and 3. Threat at 6. AI is 'O' must block at 6!
        self.game.board = ["X", " ", " ", "X", "O", " ", " ", " ", " "]
        best = get_best_move(self.game, ai_player="O")
        self.assertEqual(best, 6)

    def test_ai_prefers_winning_over_blocking(self):
        """If AI has a winning move AND opponent has a threat, AI wins immediately."""
        # Board:
        # O O . (AI can win at 2)
        # X X . (Human threatens at 5)
        # . . .
        self.game.board = ["O", "O", " ", "X", "X", " ", " ", " ", " "]
        best = get_best_move(self.game, ai_player="O")
        self.assertEqual(best, 2)

    def test_opening_move_on_empty_board(self):
        """When opening on an empty board, AI chooses center cell (index 4)."""
        best = get_best_move(self.game, ai_player="X")
        self.assertEqual(best, 4)

    def test_minimax_vs_minimax_always_draws(self):
        """Two optimal Minimax agents playing against each other must ALWAYS draw."""
        game = TicTacToe()
        current = "X"
        moves_count = 0

        while not game.is_game_over() and moves_count < 9:
            best_move = get_best_move(game, ai_player=current)
            self.assertIsNotNone(best_move)
            game.make_move(best_move, current)
            current = "O" if current == "X" else "X"
            moves_count += 1

        self.assertTrue(game.is_draw())
        self.assertIsNone(game.check_winner())

    def test_ai_never_loses_against_random_player_as_first_player(self):
        """
        Simulate 50 games where AI is 'X' (plays first) against a random player.
        The AI MUST win or draw in 100% of games (0 losses).
        """
        random.seed(42)
        for _ in range(50):
            game = TicTacToe()
            ai = "X"
            human = "O"
            current = "X"

            while not game.is_game_over():
                if current == ai:
                    move = get_best_move(game, ai_player=ai)
                    self.assertIsNotNone(move)
                    game.make_move(move, ai)
                    current = human
                else:
                    move = random.choice(game.available_moves())
                    game.make_move(move, human)
                    current = ai

            winner = game.check_winner()
            self.assertNotEqual(
                winner,
                human,
                f"AI lost as X against random player! Final board:\n{game.get_board_display()}",
            )

    def test_ai_never_loses_against_random_player_as_second_player(self):
        """
        Simulate 50 games where AI is 'O' (plays second) against a random player.
        The AI MUST win or draw in 100% of games (0 losses).
        """
        random.seed(99)
        for _ in range(50):
            game = TicTacToe()
            human = "X"
            ai = "O"
            current = "X"

            while not game.is_game_over():
                if current == human:
                    move = random.choice(game.available_moves())
                    game.make_move(move, human)
                    current = ai
                else:
                    move = get_best_move(game, ai_player=ai)
                    self.assertIsNotNone(move)
                    game.make_move(move, ai)
                    current = human

            winner = game.check_winner()
            self.assertNotEqual(
                winner,
                human,
                f"AI lost as O against random player! Final board:\n{game.get_board_display()}",
            )


if __name__ == "__main__":
    unittest.main()
