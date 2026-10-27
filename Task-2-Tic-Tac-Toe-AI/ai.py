"""
CodSoft Artificial Intelligence Internship - Task 2: Tic-Tac-Toe AI
===================================================================
Module: ai.py
Description: Implements an unbeatable Tic-Tac-Toe AI agent powered by
             the Minimax algorithm with Alpha-Beta pruning and move ordering.

===================================================================
BEGINNER-FRIENDLY EXPLANATION OF THE MINIMAX ALGORITHM
===================================================================
1. What is Minimax?
   Minimax is a classic decision-making algorithm used in two-player,
   turn-based, zero-sum games (like Tic-Tac-Toe or Chess).
   - "Maximizer" (the AI): Wants to choose moves that maximize the final score.
   - "Minimizer" (the Human): Assumed to play smartly, choosing moves that
     minimize the AI's score.

2. Scoring Terminal States:
   When the game ends:
   - If AI wins: Return (+10 - depth). Subtracting depth prioritizes quicker
     wins over prolonged ones.
   - If Opponent wins: Return (-10 + depth). Adding depth prioritizes delaying
     losses as long as possible.
   - If Draw: Return 0.

3. Recursive Decision Tree:
   For every possible move, the algorithm simulates the move and recursively
   calls itself for the other player. It backtracks until all branches are
   evaluated, propagating the optimal score up the tree.

4. Alpha-Beta Pruning & Move Ordering:
   - Alpha (α): The best value the Maximizer is guaranteed so far.
   - Beta (β): The best value the Minimizer is guaranteed so far.
   - If beta <= alpha, the opposing player would never allow this branch to be
     reached, so we can safely prune (skip) evaluating remaining sibling moves.
   - Move ordering evaluates strategic cells (center -> corners -> edges) first,
     maximizing the frequency of early cutoffs.

Result:
   Mathematically, Tic-Tac-Toe is a solved game. Minimax ensures the AI
   will NEVER lose. The outcome is either an AI win or a draw.
===================================================================
"""

import math
from typing import List, Optional, Tuple

from game import TicTacToe


# Move ordering preference: Center is most strategic, then corners, then edges.
PREFERRED_MOVE_ORDER: List[int] = [4, 0, 2, 6, 8, 1, 3, 5, 7]


def order_moves(available: List[int]) -> List[int]:
    """Sort available moves by strategic priority to improve alpha-beta pruning."""
    return sorted(available, key=lambda move: PREFERRED_MOVE_ORDER.index(move))


def minimax(
    game: TicTacToe,
    depth: int,
    is_maximizing: bool,
    ai_player: str,
    human_player: str,
    alpha: float = -math.inf,
    beta: float = math.inf,
) -> int:
    """
    Minimax recursive evaluation with Alpha-Beta pruning.

    Args:
        game: Current TicTacToe board instance.
        depth: Current recursion depth in the search tree.
        is_maximizing: True if evaluating AI's turn, False for human's turn.
        ai_player: Marker symbol for AI ('X' or 'O').
        human_player: Marker symbol for Human ('X' or 'O').
        alpha: Current alpha cutoff threshold.
        beta: Current beta cutoff threshold.

    Returns:
        The heuristic/terminal score of the evaluated game state.
    """
    winner = game.check_winner()

    # Terminal state evaluations
    if winner == ai_player:
        # Prefer faster wins (higher score)
        return 10 - depth
    if winner == human_player:
        # Prefer delayed losses (less negative score)
        return -10 + depth
    if game.is_board_full():
        # Draw game
        return 0

    available = order_moves(game.available_moves())

    if is_maximizing:
        max_eval = -math.inf
        for move in available:
            game.make_move(move, ai_player)
            eval_score = minimax(
                game, depth + 1, False, ai_player, human_player, alpha, beta
            )
            game.undo_move(move)

            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        return int(max_eval)
    else:
        min_eval = math.inf
        for move in available:
            game.make_move(move, human_player)
            eval_score = minimax(
                game, depth + 1, True, ai_player, human_player, alpha, beta
            )
            game.undo_move(move)

            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        return int(min_eval)


def get_best_move(game: TicTacToe, ai_player: str) -> Optional[int]:
    """
    Determine the optimal move for the AI using Minimax with Alpha-Beta pruning.

    Args:
        game: Current TicTacToe board instance.
        ai_player: Marker symbol representing the AI ('X' or 'O').

    Returns:
        The optimal 0-based cell index to play, or None if no moves remain.
    """
    available = game.available_moves()
    if not available:
        return None

    # Derive human player marker
    human_player = (
        TicTacToe.PLAYER_O if ai_player == TicTacToe.PLAYER_X else TicTacToe.PLAYER_X
    )

    # Shortcut 1: If board is completely empty, center is always an optimal opening
    if len(available) == 9:
        return 4

    # Shortcut 2: Immediate winning move check (depth 1 optimization)
    for move in available:
        game.make_move(move, ai_player)
        if game.check_winner() == ai_player:
            game.undo_move(move)
            return move
        game.undo_move(move)

    # Shortcut 3: Immediate human blocking move check (depth 1 optimization)
    for move in available:
        game.make_move(move, human_player)
        if game.check_winner() == human_player:
            game.undo_move(move)
            return move
        game.undo_move(move)

    # Full Minimax evaluation for remaining positions
    best_score = -math.inf
    best_move = available[0]
    alpha = -math.inf
    beta = math.inf

    ordered = order_moves(available)
    for move in ordered:
        game.make_move(move, ai_player)
        score = minimax(
            game,
            depth=0,
            is_maximizing=False,
            ai_player=ai_player,
            human_player=human_player,
            alpha=alpha,
            beta=beta,
        )
        game.undo_move(move)

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)

    return best_move
