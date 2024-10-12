import gymnasium as gym
from gymnasium import spaces
import re
import numpy as np

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

        self.rewards = cn.Reward if reward_values is None else reward_values

        # Load full word list
        self.word_file = word_file
        with open(self.word_file) as f:
            full_word_list = f.readlines()
            self.full_word_list = np.array([word.strip().upper() for word in full_word_list])

        # Gym Environment Attributes
        self.reset()        # Initialize game state
        self.observation_space = spaces.Dict({
            'colors': spaces.MultiDiscrete([len(cn.Color)] * self.num_words),
            'words': spaces.Tuple([spaces.Text(self.MAX_TEXT_LENGTH)] * self.num_words),
            'guessed': spaces.MultiBinary(self.num_words),
            'turn': spaces.MultiDiscrete([len(cn.Team), len(cn.Role)]),
            'hint': spaces.Tuple((spaces.Text(self.MAX_TEXT_LENGTH), 
                                    spaces.Box(low=0, high=np.inf))),
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
            guess_mask = self.state['words'] == guess
            self.state['guessed'][guess_mask] = True
            guess_color = self.state['colors'][guess_mask][0]
            other_team = cn.Team.BLUE if team == cn.Team.RED else cn.Team.RED
            if guess_color.name == team.name:           # Correct guess
                info['result'] = f"Correct guess: {guess}."
                print("Hint:", self.state['hint'])
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
            elif guess_color == cn.Color.NEUTRAL:       # Neutral guess
                info['result'] = f"Neutral guess: {guess}. Turnover."
                self.turnover()
                reward_n[(team, cn.Role.OPERATIVE)] = self.rewards.NEUTRAL_GUESS_OPERATIVE.value
                reward_n[(team, cn.Role.SPYMASTER)] = self.rewards.NEUTRAL_GUESS_SPYMASTER.value
                winner = None
            elif guess_color.name == other_team.name:   # Wrong guess
                info['result'] = f"Wrong guess: {guess}. Turnover."
                self.turnover()
                reward_n[(team, cn.Role.OPERATIVE)] = self.rewards.WRONG_GUESS_OPERATIVE.value
                reward_n[(team, cn.Role.SPYMASTER)] = self.rewards.WRONG_GUESS_SPYMASTER.value
                winner = self.check_winner()
            elif guess_color == cn.Color.ASSASSIN:      # Assassin guess
                info['result'] = f"Assassin guess: {guess}. Game over."
                reward_n[(team, cn.Role.OPERATIVE)] = self.rewards.ASSASSIN_GUESS_OPERATIVE.value
                reward_n[(team, cn.Role.SPYMASTER)] = self.rewards.ASSASSIN_GUESS_SPYMASTER.value
                done = True
                winner = other_team
            else:
                raise ValueError(f"Invalid board state value: {guess_color}.")
            
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
        words = np.random.choice(self.full_word_list, self.num_words)
        colors = np.array(
            [cn.Color.RED] * self.num_red +
            [cn.Color.BLUE] * self.num_blue +
            [cn.Color.NEUTRAL] * self.num_neutral +
            [cn.Color.ASSASSIN] * self.num_assassin
        )
        np.random.shuffle(colors)
        self.state = {
            'words': words,
            'colors': colors,
            'guessed': np.zeros(self.num_words, dtype=bool),
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
            reward += self.rewards.HINT_UNPARSEABLE.value
            return None, reward, result
        elif len(hints) > 1:
            result = f"Invalid hint: {hint}. Only one hint word and count allowed."
            reward += self.rewards.HINT_MULTIPLE.value
            return None, reward, result
        reward += self.rewards.HINT_PARSEABLE.value

        word, count = hints[0]

        # Validate count
        if not count.isdigit() or int(count) < 0:
            result = f"Invalid hint: {hint}. Count must be a non-negative integer."
            reward += self.rewards.HINT_INVALID_COUNT.value
            return None, reward, result
        count = int(count)
        reward += self.rewards.HINT_VALID_COUNT.value
        
        # Validate word
        word = word.strip().upper()
        if np.isin(word.split(), self.state['words'][~self.state['guessed']]):
            result = f"Invalid hint: {hint}. Hint word or subword is on the board."
            reward += self.rewards.HINT_WORD_ON_BOARD.value
            return None, reward, result
        reward += self.rewards.HINT_VALID_WORD.value

        hint = (word, count)
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
        elif self.state['guessed'][self.state['words'] == guess]:
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
        remaining = self.state['colors'][self.state['guessed']]
        if np.all(remaining != cn.Color.RED):
            return cn.Team.RED
        elif np.all(remaining != cn.Color.BLUE):
            return cn.Team.BLUE
        return None
            
            

    

        
        