"""
CodSoft Artificial Intelligence Internship - Task 2: Tic-Tac-Toe AI
===================================================================
Module: main.py
Description: Interactive command-line interface for human vs AI gameplay.
===================================================================
"""

import sys
from typing import Tuple

from ai import get_best_move
from game import TicTacToe


def print_banner() -> None:
    """Print the game intro banner and board coordinate guide."""
    border = "=" * 60
    print(f"\n{border}")
    print("      CodSoft Artificial Intelligence Internship")
    print("             Task 2: Tic-Tac-Toe AI Agent        ")
    print(f"{border}")
    print("  * Architecture: Unbeatable Minimax with Alpha-Beta Pruning")
    print("  * Coordinates: Enter numbers 1 through 9 to choose a cell:")
    print("       1 | 2 | 3")
    print("      ---+---+---")
    print("       4 | 5 | 6")
    print("      ---+---+---")
    print("       7 | 8 | 9")
    print(f"{border}\n")


def get_player_choice() -> Tuple[str, str]:
    """
    Prompt the human player to select their marker (X or O).
    Returns (human_marker, ai_marker).
    """
    while True:
        try:
            choice = (
                input("Choose your marker ('X' plays first, 'O' plays second) [X/O, default: X]: ")
                .strip()
                .upper()
            )
            if choice == "" or choice == "X":
                return TicTacToe.PLAYER_X, TicTacToe.PLAYER_O
            elif choice == "O":
                return TicTacToe.PLAYER_O, TicTacToe.PLAYER_X
            print("Invalid selection. Please enter 'X' or 'O'.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting setup.")
            sys.exit(0)


def get_human_move(game: TicTacToe) -> int:
    """
    Prompt the human user for a valid 1-9 move and validate input.
    Returns 0-based cell index.
    """
    while True:
        try:
            user_input = input("Enter your move (1-9) or 'q' to quit: ").strip().lower()
            if user_input in {"q", "quit", "exit"}:
                print("\nQuitting current game. Goodbye!")
                sys.exit(0)

            if not user_input.isdigit():
                print("Invalid input! Please enter a number between 1 and 9.")
                continue

            cell_number = int(user_input)
            if not (1 <= cell_number <= 9):
                print("Out of range! Please choose a cell from 1 to 9.")
                continue

            index = cell_number - 1
            if not game.is_valid_move(index):
                print(f"Cell {cell_number} is already occupied! Choose an open cell.")
                continue

            return index
        except (KeyboardInterrupt, EOFError):
            print("\nGame aborted by user.")
            sys.exit(0)


def play_round() -> None:
    """Run a single round of Tic-Tac-Toe between human and AI."""
    game = TicTacToe()
    human_marker, ai_marker = get_player_choice()

    print(f"\nYou are playing as [{human_marker}]. AI is playing as [{ai_marker}].")
    print("Let the game begin!\n")

    current_player = TicTacToe.PLAYER_X  # 'X' always goes first

    while not game.is_game_over():
        game.print_board()

        if current_player == human_marker:
            print(f"Your turn [{human_marker}]:")
            move = get_human_move(game)
            game.make_move(move, human_marker)
            current_player = ai_marker
        else:
            print(f"AI [{ai_marker}] is thinking...")
            best_move = get_best_move(game, ai_marker)
            if best_move is not None:
                game.make_move(best_move, ai_marker)
                print(f"AI chose cell {best_move + 1}.\n")
            current_player = human_marker

    # Show final board state
    game.print_board()

    # Determine match result
    winner = game.check_winner()
    border = "-" * 40
    print(f"\n{border}")
    if winner == human_marker:
        print(f"  Result: CONGRATULATIONS! You won as [{human_marker}]!")
    elif winner == ai_marker:
        print(f"  Result: GAME OVER! The AI [{ai_marker}] won!")
    else:
        print("  Result: IT'S A DRAW! Well played!")
    print(f"{border}\n")


def main() -> None:
    """Main interactive execution loop supporting replays."""
    print_banner()

    while True:
        play_round()

        try:
            again = (
                input("Would you like to play another game? (y/n, default: y): ")
                .strip()
                .lower()
            )
            if again in {"n", "no", "exit", "quit"}:
                print("\nThank you for playing Tic-Tac-Toe AI! Goodbye.\n")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting game. Goodbye!")
            break


if __name__ == "__main__":
    main()
