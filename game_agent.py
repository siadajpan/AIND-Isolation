
from math import sqrt, floor

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def moves_no(game, p):
    """ Calculate amount of moves that the player has """
    return len(game.get_legal_moves(p))

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Score is a linear combination of custom score 2 and 3

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    
    if game.is_loser(player):
        return float('-inf')
    if game.is_winner(player):
        return float('inf')
    
    alpha = 1.0
    progress = len(game.get_blank_spaces()) / (game.width * game.height)
    
    return progress * custom_score_2(game, player) + \
                alpha * (1 - progress) * custom_score_3(game, player)
    
    
def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    The heuristic gives the best score to the middle positions and is getting
    smaller close to the walls and the edges. 
    It's calculated using half sphere that rises above the playfield.
    Score is the height of the point that sphere above current position.
    Sphere has radius of half of the length of field (or quarter of the average
    of width and height for non-square fields).
    
    That gives nice near equal scores for positions close to the centre, 
    and score 0 to almost all the walls values and values near the corner.
    

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    my_loc = game.get_player_location(player)
    score = 1.
    try: 
        score = player.square_lookup_score2[str(my_loc[0]) + str(my_loc[1])]
    except AttributeError: # if player is not inheriting after Isolation Player
        score = calculate_sphere_z (my_loc[0], my_loc[1], 
                                    (game.width + game.height)/4,
                                    gamma = 1.)
        
    return score

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    gamma = 2
    score = moves_no(game, player) - gamma * moves_no(game, game.get_opponent(player))
    
    return float(score)

def calculate_sphere_z(x, y, r, gamma):
    dist_cntr = sqrt((x - r)**2 + (y - r)**2)

    if dist_cntr <= gamma * r:
        score = sqrt((gamma * r)**2 - dist_cntr**2)
    else:
        score = 0.
        
    return score

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.square_lookup_score2 = self.init_lookup_score2(7, 1.2)
    
    def init_lookup_score2(self, board_dimension, gamma=1.):
        """Create lookup table to custom score 2 as a dictionary.
        The keys of the dictionary are concatenated position of row and column, 
        and the value is the z value of the sphere centred in the centre of 
        the board.
        
        Parameters
        ----------
        
        board_dimension : int
            Specifies width of square board
            
        gamma: float (optional)
            Multiplier for the sphere radius. By changing it to smaller value
            we let positions near the wall to be 0. Increasing this number
            will bring less penalty for positions closer to the wall or corner
        """
        
        r = floor(board_dimension / 2)
        lookup_dict = {}
        
        for x in range(board_dimension):
            for y in range(board_dimension):
                # key is a string concatenation of row and column position
                lookup_dict[str(x) + str(y)] = calculate_sphere_z(x, y, r, gamma)
                
        return lookup_dict


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def min_value(self, game, depth):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0 or len(game.get_legal_moves()) == 0:
            return self.score(game, self)

        value = float("inf")
        for move in game.get_legal_moves():
            value = min(value, self.max_value(game.forecast_move(move), depth - 1))
        return value

    def max_value(self, game, depth):

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 0 or len(game.get_legal_moves()) == 0:
            return self.score(game, self)

        value = float("-inf")
        for move in game.get_legal_moves():
            value = max(value, self.min_value(game.forecast_move(move), depth - 1))
        return value

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        # initialize the best score and best move
        best_score = float("-inf")
        best_move = (-1, -1)

        # for each legal move, implement recursive search
        for move in legal_moves:
            score = self.min_value(game.forecast_move(move), depth-1)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move



class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout

        best_move = (-1, -1)
        depth = 1

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while time_left() > self.TIMER_THRESHOLD:
                next_move = self.alphabeta(game, depth)
                if not next_move:
                    return best_move
                else:
                    # If tree is successfully searched, update best_move
                    best_move = next_move
                depth += 1
        except SearchTimeout:
            return best_move 

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        def max_value(game, depth, alpha, beta):
            # raise the Timeout if time is reached
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            # get the score if depth limit is reached
            elif depth == 0:
                return self.score(game, self)
            
            # initialize to lowest possible number, only higher will be saved
            v = float('-inf')
            
            if game.get_legal_moves():
                for action in game.get_legal_moves():
                    v = max(v, min_value(game.forecast_move(action), depth - 1, 
                                         alpha, beta))
                    
                    # save score if higher than current max
                    if v >= beta: 
                        return v
                    
                    # update alpha if not returned
                    alpha = max(alpha, v)
                    
            return v
        
        def min_value(game, depth, alpha, beta):
            # raise the Timeout if time is reached
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
                
            # get the score if depth limit is reached
            if depth == 0:
                return self.score(game, self)
            
            # initialize to highest possible number, only lower will be saved
            v = float('inf')
            if game.get_legal_moves():
                for action in game.get_legal_moves():
                    v = min(v, max_value(game.forecast_move(action), depth - 1,
                                         alpha, beta))
                    
                    # save score if lower than current min
                    if v <= alpha: 
                        return v
                    
                    # update beta if not returned
                    beta = min(beta, v)
            return v

        # Set max value to the -inf.  This function searches for max value 
        best_score = float("-inf")
        # Set best move
        my_moves = game.get_legal_moves()        
        
        try:
            best_move = game.get_legal_moves()[0]
        except (IndexError, ValueError) as e:
            best_move = (-1, -1)
            
        for move in my_moves:
            possible_game = game.forecast_move(move)
            score = min_value(possible_game, depth-1, alpha, beta)
            
            # update best move and score if found better
            if score > best_score:
                best_move, best_score = move, score
            
            # prune move if score is better than beta
            if best_score >= beta:
                break

            # Update Alpha
            alpha = max(alpha, best_score)
            
        return best_move