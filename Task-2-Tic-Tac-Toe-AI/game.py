"""
CodSoft Artificial Intelligence Internship - Task 2: Tic-Tac-Toe AI
===================================================================
Module: game.py
Description: Encapsulates the Tic-Tac-Toe board state, move execution,
             rules validation, terminal condition checks (win/draw),
             and formatted board rendering.
===================================================================
"""

from typing import List, Optional, Tuple


class TicTacToe:
    """Represents a standard 3x3 Tic-Tac-Toe game board."""

    EMPTY: str = " "
    PLAYER_X: str = "X"
    PLAYER_O: str = "O"

    # All 8 possible winning combinations (rows, columns, diagonals)
    WINNING_COMBINATIONS: List[Tuple[int, int, int]] = [
        (0, 1, 2),  # Top row
        (3, 4, 5),  # Middle row
        (6, 7, 8),  # Bottom row
        (0, 3, 6),  # Left column
        (1, 4, 7),  # Middle column
        (2, 5, 8),  # Right column
        (0, 4, 8),  # Main diagonal (\)
        (2, 4, 6),  # Anti diagonal (/)
    ]

    def __init__(self) -> None:
        """Initialize an empty 3x3 board."""
        self.board: List[str] = [self.EMPTY] * 9

    def reset(self) -> None:
        """Reset the board to the initial empty state."""
        self.board = [self.EMPTY] * 9

    def clone(self) -> "TicTacToe":
        """Return an independent copy of the current board for simulations."""
        new_game = TicTacToe()
        new_game.board = list(self.board)
        return new_game

    def available_moves(self) -> List[int]:
        """Return a list of 0-based indices for all unoccupied cells."""
        return [i for i, cell in enumerate(self.board) if cell == self.EMPTY]

    def is_valid_move(self, index: int) -> bool:
        """Check whether a 0-based index is within bounds and unoccupied."""
        return 0 <= index < 9 and self.board[index] == self.EMPTY

    def make_move(self, index: int, player: str) -> bool:
        """
        Place the player's marker at the given 0-based index.

        Returns:
            True if the move was successfully made, False otherwise.
        """
        if self.is_valid_move(index):
            self.board[index] = player
            return True
        return False

    def undo_move(self, index: int) -> None:
        """Revert a move by setting the specified cell back to empty."""
        if 0 <= index < 9:
            self.board[index] = self.EMPTY

    def check_winner(self) -> Optional[str]:
        """
        Check if any player has won the game.

        Returns:
            'X' or 'O' if there is a winner, or None if no winner yet.
        """
        for a, b, c in self.WINNING_COMBINATIONS:
            if (
                self.board[a] != self.EMPTY
                and self.board[a] == self.board[b] == self.board[c]
            ):
                return self.board[a]
        return None

    def is_board_full(self) -> bool:
        """Check if all 9 cells are occupied."""
        return self.EMPTY not in self.board

    def is_draw(self) -> bool:
        """Return True if the board is completely full and there is no winner."""
        return self.is_board_full() and self.check_winner() is None

    def is_game_over(self) -> bool:
        """Check if the game has ended either by win or by draw."""
        return self.check_winner() is not None or self.is_board_full()

    def get_board_display(self) -> str:
        """
        Return a formatted string representing the current board state.
        Empty cells display their 1-9 coordinate numbers to guide player input.
        """
        display_cells = [
            str(i + 1) if cell == self.EMPTY else cell
            for i, cell in enumerate(self.board)
        ]
        divider = "  ---+---+---"
        row1 = f"   {display_cells[0]} | {display_cells[1]} | {display_cells[2]} "
        row2 = f"   {display_cells[3]} | {display_cells[4]} | {display_cells[5]} "
        row3 = f"   {display_cells[6]} | {display_cells[7]} | {display_cells[8]} "
        return f"\n{row1}\n{divider}\n{row2}\n{divider}\n{row3}\n"

    def print_board(self) -> None:
        """Print the current formatted board to stdout."""
        print(self.get_board_display())
