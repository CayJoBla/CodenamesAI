import random
import math
import gymnasium as gym
from gymnasium import spaces
import re

from . import constants as cn


class CodenamesEnv(gym.Env):

    MAX_TEXT_LENGTH = 30

    def __init__(
        self, 
        board_shape=(5, 5),
        num_red=9, 
        num_blue=8,
        num_assassin=1,
        reward_values=None,
        word_file='codenames/word_lists/codenames.txt',
    ):
        # Define game board attributes
        self.board_shape = board_shape
        self.num_words = board_shape[0] * board_shape[1]
        self.num_red = num_red
        self.num_blue = num_blue
        self.num_assassin = num_assassin
        self.num_neutral = self.num_words - (num_red + num_blue + num_assassin)
        if self.num_neutral < 0:
            raise ValueError("Invalid number of words for board size.")

        if reward_values is None:
            self.rewards = cn.Reward

        # Load full word list
        self.word_file = word_file
        with open(self.word_file) as f:
            full_word_list = f.readlines()
            self.full_word_list = [word.strip().upper() for word in full_word_list]


        # Gym Environment Attributes
        self.reset()        # Initialize game state
        self.observation_space = spaces.Dict({
            'board': spaces.MultiDiscrete([len(cn.Color)] * self.num_words),
            'words': spaces.Tuple([spaces.Text(self.MAX_TEXT_LENGTH)] * self.num_words),
            'guessed': spaces.MultiBinary(self.num_words),
            'turn': spaces.MultiDiscrete([len(cn.Team), len(cn.Role)]),
            'hint': spaces.Tuple((spaces.Text(self.MAX_TEXT_LENGTH), 
                                    spaces.Box(low=0, high=math.inf))),
        })
        self.action_space = spaces.Text(self.MAX_TEXT_LENGTH)

    def step(self, action):
        # Default return values
        info = {}       
        done = False
        reward_n = {
            (cn.Team.RED, cn.Role.SPYMASTER): self.rewards.IDLE.value,
            (cn.Team.RED, cn.Role.OPERATIVE): self.rewards.IDLE.value,
            (cn.Team.BLUE, cn.Role.SPYMASTER): self.rewards.IDLE.value,
            (cn.Team.BLUE, cn.Role.OPERATIVE): self.rewards.IDLE.value,
        }
        team, role = self.state['turn']

        # Spymaster's turn
        if role == cn.Role.SPYMASTER:
            try:
                hint, reward, result = self.parse_hint(action)  
            except Exception as e:
                hint = None
                result = f"Invalid hint: {action}. Game over."
                reward = self.rewards.HINT_UNPARSEABLE.value
                info['error'] = f"{type(e).__name__}: {e}"
            if hint is None:
                info['result'] = result
                reward_n[(team, role)] += reward
                done = True
                return self.state, reward_n, done, info
            self.state['hint'] = hint
            self.state['turn'] = (team, cn.Role.OPERATIVE)
            info['result'] = result
            reward_n[(team, role)] = reward

        # Operative's (Guesser's) turn
        elif role == cn.Role.OPERATIVE:
            try:
                guess, reward, result = self.parse_guess(action)
            except Exception as e:
                guess = None
                result = f"Invalid guess: {action}. Game over."
                reward = self.rewards.GUESS_UNPARSEABLE.value
                info['error'] = f"{type(e).__name__}: {e}"
            if guess is None:
                info['result'] = result
                reward_n[(team, role)] += reward
                done = True
                return self.state, reward_n, done, info
            
            # Pass turn and abstain from guessing   
            if guess == '':  
                info['result'] = result
                reward_n[(team, role)] += reward
                self.turnover()
                return self.state, reward_n, done, info

            # Make a guess
            guess_idx = self.state['words'].index(guess)
            self.state['guessed'][guess_idx] = True
            guess_value = self.state['board'][guess_idx]
            other_team = cn.Team.BLUE if team == cn.Team.RED else cn.Team.RED
            if guess_value == team:             # Correct guess
                info['result'] = f"Correct guess: {guess}."
                if self.state['hint'][1] > 0:
                    self.state['hint'] = (self.state['hint'][0], self.state['hint'][1] - 1)
                else:
                    info['result'] += " Out of guesses. Turnover."
                    self.turnover()
                reward_n[(team, cn.Role.OPERATIVE)] = self.rewards.CORRECT_GUESS_OPERATIVE.value
                reward_n[(team, cn.Role.SPYMASTER)] = self.rewards.CORRECT_GUESS_SPYMASTER.value
                if self.state['hint'][1] == 0:
                    reward_n[(team, cn.Role.OPERATIVE)] += self.rewards.ALL_GUESSES_BONUS_OPERATIVE.value
                    reward_n[(team, cn.Role.SPYMASTER)] += self.rewards.ALL_GUESSES_BONUS_SPYMASTER.value
                winner = self.check_winner()
            elif guess_value == cn.Color.NEUTRAL:   # Neutral guess
                info['result'] = f"Neutral guess: {guess}. Turnover."
                self.turnover()
                reward_n[(team, cn.Role.OPERATIVE)] = self.rewards.NEUTRAL_GUESS_OPERATIVE.value
                reward_n[(team, cn.Role.SPYMASTER)] = self.rewards.NEUTRAL_GUESS_SPYMASTER.value
                winner = None
            elif guess_value == other_team:     # Wrong guess
                info['result'] = f"Wrong guess: {guess}. Turnover."
                self.turnover()
                reward_n[(team, cn.Role.OPERATIVE)] = self.rewards.WRONG_GUESS_OPERATIVE.value
                reward_n[(team, cn.Role.SPYMASTER)] = self.rewards.WRONG_GUESS_SPYMASTER.value
                winner = self.check_winner()
            elif guess_value == cn.Color.ASSASSIN:  # Assassin guess
                info['result'] = f"Assassin guess: {guess}. Game over."
                reward_n[(team, cn.Role.OPERATIVE)] = self.rewards.ASSASSIN_GUESS_OPERATIVE.value
                reward_n[(team, cn.Role.SPYMASTER)] = self.rewards.ASSASSIN_GUESS_SPYMASTER.value
                done = True
                winner = other_team
            else:
                raise ValueError("Invalid board state value.")
            
            if winner is not None:
                not_winner = cn.Team.BLUE if winner == cn.Team.RED else cn.Team.RED
                winner_name = "Red" if winner == cn.Team.RED else "Blue"
                info['result'] += f" {winner_name} wins!"
                done = True
                reward_n[(winner, cn.Role.SPYMASTER)] += self.rewards.WIN.value
                reward_n[(winner, cn.Role.OPERATIVE)] += self.rewards.WIN.value
                reward_n[(not_winner, cn.Role.SPYMASTER)] += self.rewards.LOSS.value
                reward_n[(not_winner, cn.Role.OPERATIVE)] += self.rewards.LOSS.value
                
        else:
            raise ValueError("Invalid role in turn.")

        return self.state, reward_n, done, info

    def reset(self):
        # Create board words and team assignments
        board = (
            [cn.Color.RED] * self.num_red +
            [cn.Color.BLUE] * self.num_blue +
            [cn.Color.NEUTRAL] * self.num_neutral +
            [cn.Color.ASSASSIN] * self.num_assassin
        )
        random.shuffle(board)
        words = random.sample(self.full_word_list, self.num_words)

        self.state = {
            'board': board,
            'words': words,
            'guessed': [False] * self.num_words,
            'turn': (cn.Team.RED, cn.Role.SPYMASTER),
        }
        return self.state

    def render(self):
        # TODO: Implement render method to display game state
        return None

    def parse_hint(self, hint):
        # Check for valid format
        reward = 0
        hints = re.findall(r"([a-zA-Z ]+) (\d+)", hint.strip())
        if len(hints) == 0:
            result = f"Invalid hint: {hint}. Hint must be in the format 'word count'."
            reward += self.rewards.HINT_INPARSEABLE.value
            return None, reward, result
        elif len(hints) > 1:
            result = f"Invalid hint: {hint}. Only one hint word and count allowed."
            reward += self.rewards.HINT_MULTIPLE.value
            return None, reward, result
        reward += self.rewards.HINT_PARSEABLE.value

        hint = hints[0]
        word, count = hint

        # Validate count
        if not count.isdigit() or int(count) < 0:
            result = f"Invalid hint: {hint}. Count must be a non-negative integer."
            reward += self.rewards.HINT_INVALID_COUNT.value
            return None, reward, result
        reward += self.rewards.HINT_VALID_COUNT.value
        
        # Validate word
        word = word.strip().upper()
        subwords = word.split()
        for i, game_word in enumerate(self.state['words']):
            if self.state['guessed'][i]:
                continue
            if game_word in subwords:
                result = f"Invalid hint: {hint}. Hint word is on the board."
                reward += self.rewards.HINT_WORD_ON_BOARD.value
                return None, reward, result
        reward += self.rewards.HINT_VALID_WORD.value

        return hint, reward, f"Valid hint: {hint}."

    def parse_guess(self, guess):
        # Check for valid format
        reward = 0
        guesses = re.findall(r"[a-zA-Z ]+", guess.strip())
        if len(guesses) == 0:
            result = "No guess made. Turnover."
            reward += self.rewards.GUESS_SKIP.value
            return '', reward, result
        elif len(guesses) > 1:
            result = f"Invalid guess: {guess}. Only one guess is allowed."
            reward += self.rewards.GUESS_MULTIPLE.value
            return None, reward, result
        reward += self.rewards.GUESS_PARSEABLE.value

        guess = guesses[0].strip().upper()
        if not guess in self.state['words']:
            result = f"Invalid guess: {guess}. Guess is not a word on the board."
            reward += self.rewards.GUESS_NOT_ON_BOARD.value
            return None, reward, result
        
        index = self.state['words'].index(guess)
        if self.state['guessed'][index]:
            result = f"Invalid guess: {guess}. Guess has already been made."
            reward += self.rewards.GUESS_ALREADY_MADE.value
            return None, reward, result

        reward += self.rewards.GUESS_VALID.value
        return guess, reward, f"Valid guess: {guess}."

    def turnover(self):
        other_team = cn.Team.BLUE if self.state['turn'][0] == cn.Team.RED else cn.Team.RED
        self.state['turn'] = (other_team, cn.Role.SPYMASTER)
        self.state['hint'] = None

    def check_winner(self):
        remaining = {
            cn.Color.RED: 0, 
            cn.Color.BLUE: 0, 
            cn.Color.NEUTRAL: 0, 
            cn.Color.ASSASSIN: 0
        }
        for i, word in enumerate(self.state['words']):
            if self.state['guessed'][i]:
                continue
            remaining[self.state['board'][i]] += 1
            if remaining[cn.Color.RED] > 0 and remaining[cn.Color.BLUE] > 0:
                return None
        if remaining[cn.Color.RED] == 0:
            return cn.Team.RED
        elif remaining[cn.Color.BLUE] == 0:
            return cn.Team.BLUE
        return None
            
            

    

        
        