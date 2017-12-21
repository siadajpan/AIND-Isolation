"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent
import sample_players
from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = game_agent.MinimaxPlayer()
        self.player2 = sample_players.RandomPlayer()
        self.game = isolation.Board(self.player1, self.player2)

    def test_game(self):
        self.game.apply_move((2, 3))
        self.game.apply_move((0, 5))
        print(self.game.to_string())
        assert(self.player1 == self.game.active_player)
        print(self.game.get_legal_moves())

        new_game = self.game.forecast_move((1, 1))
        assert(self.game != new_game)
        assert(self.game.to_string() != new_game.to_string())
        
        print("\nOld state:\n{}".format(self.game.to_string()))
        print("\nNew state:\n{}".format(new_game.to_string()))
        
        winner, history, outcome = self.game.play()
        print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
        print(self.game.to_string())
        print("Move history:\n{!s}".format(history))
        
    def test_minmax(self):
        self.game.apply_move((6, 6))
        self.game.apply_move((0, 5))
        self.game.apply_move((5, 4))
        self.game.apply_move((2, 4))
        self.game.apply_move((4, 2))
        self.game.apply_move((3, 6))
        self.game.apply_move((2, 3))
        self.game.apply_move((5, 5))
        self.game.apply_move((1, 5))
        self.game.apply_move((4, 3))
        self.game.apply_move((2, 7))
        self.game.apply_move((2, 2))
        self.game.apply_move((3, 5))
        self.game.apply_move((4, 0))
        self.game.apply_move((1, 4))        
        self.game.apply_move((2, 3))
        self.game.apply_move((0, 2))
        self.game.apply_move((4, 4))
        self.game.apply_move((2, 1))
        self.game.apply_move((6, 3))
        self.game.apply_move((0, 0))
        self.game.apply_move((5, 1))
        self.game.apply_move((1, 2))
        self.game.apply_move((2, 3))
        self.game.apply_move((2, 0))
        self.game.apply_move((1, 3))
        
        print(self.game.to_string())
        winner, history, outcome = self.game.play()
        print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
        
        
if __name__ == '__main__':
    unittest.main()