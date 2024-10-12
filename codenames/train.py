import os
import numpy as np
from enum import Enum

from .gym import CodenamesEnv
from .agents import Human, Dummy, Spymaster, Operative
from . import constants as cn

DEFAULT_AI_MODEL = "gpt2"

class TrainReward(Enum):
    IDLE = 0
    HINT_UNPARSEABLE = -10
    HINT_MULTIPLE = -5
    HINT_PARSEABLE = 1
    HINT_INVALID_COUNT = -3
    HINT_VALID_COUNT = 1
    HINT_WORD_ON_BOARD = -3
    HINT_VALID_WORD = 1
    GUESS_UNPARSEABLE = -10
    GUESS_SKIP = -1
    GUESS_MULTIPLE = -5
    GUESS_PARSEABLE = 1
    GUESS_NOT_ON_BOARD = -3
    GUESS_ALREADY_MADE = -3
    GUESS_VALID = 1
    CORRECT_GUESS_OPERATIVE = 1
    CORRECT_GUESS_SPYMASTER = 1
    ALL_GUESSES_BONUS_OPERATIVE = 1
    ALL_GUESSES_BONUS_SPYMASTER = 1
    NEUTRAL_GUESS_OPERATIVE = -3
    NEUTRAL_GUESS_SPYMASTER = -3
    WRONG_GUESS_OPERATIVE = -5
    WRONG_GUESS_SPYMASTER = -5
    ASSASSIN_GUESS_OPERATIVE = -10
    ASSASSIN_GUESS_SPYMASTER = -10
    WIN = 0     # Remove the WIN and LOSS rewards
    LOSS = 0

def codenames(
    spymaster,
    operative,
    board_shape=(5, 5),
    num_red=9,
    num_blue=8,
    num_assassin=1,
    word_file='codenames/word_lists/codenames.txt',
    allow_human_player=False,
    do_train=False,
    num_games=1,
    save_dir=None,
):
    # Initialize the Codenames game environment
    env = CodenamesEnv(
        board_shape=board_shape,
        num_red=num_red, 
        num_blue=num_blue,
        num_assassin=num_assassin,
        word_file=word_file,
        reward_values=TrainReward
    )
    
    team = cn.Team.RED

    # Initialize the agents
    role = cn.Role.SPYMASTER
    if spymaster is None:
        spymaster = Human(team, role) if allow_human_player else Dummy(team, role)
    elif isinstance(spymaster, str):
        spymaster = Spymaster(team, model_name_or_path=spymaster)
    elif not isinstance(spymaster, Spymaster):
        raise ValueError("Invalid spymaster argument.")
    
    role = cn.Role.OPERATIVE
    if operative is None:
        operative = Human(team, role) if allow_human_player else Dummy(team, role) 
    elif isinstance(operative, str):
        operative = Operative(team, model_name_or_path=operative)
    elif not isinstance(spymaster, Operative):
        raise ValueError("Invalid operative argument.")

    # Run the Codenames game
    for game in range(num_games):
        env.reset()
        done = False
        while not done:
            # Get team
            team = env.state['turn'][0]
            spymaster.team = team
            operative.team = team

            # Spymaster's turn
            assert env.state['turn'][1] == cn.Role.SPYMASTER
            action = spymaster.get_action(env.state)
            _, reward_n, done, info = env.step(action)
            spymaster_reward = reward_n[(team, cn.Role.SPYMASTER)]
            print("Spymaster's Action:", action)
            print("Current Reward:", spymaster_reward)
            print("Info:", info)

            # Operative's turn
            while not done and team == env.state['turn'][0]:
                assert env.state['turn'][1] == cn.Role.OPERATIVE
                obs_state = {k: v for k, v in env.state.items() if k != 'colors'}
                action = operative.get_action(obs_state)
                _, reward_n, done, info = env.step(action)
                spymaster_reward += reward_n[(team, cn.Role.SPYMASTER)]
                if do_train:
                    operative.step(reward_n[(team, cn.Role.OPERATIVE)])
                print("Operative's Action:", action)
                print("Operative's Reward:", reward_n[(team, cn.Role.OPERATIVE)])
                print("Spymaster's Final Reward:", spymaster_reward)
                print("Info:", info)

            if do_train:
                spymaster.step(spymaster_reward)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        spymaster.save(os.path.join(save_dir, f"{spymaster.role.name}"))
        operative.save(os.path.join(save_dir, f"{operative.role.name}"))

    return spymaster, operative


if __name__ == "__main__":
    codenames(
        spymaster = "meta-llama/Llama-3.2-1B-Instruct",
        operative = "meta-llama/Llama-3.2-1B-Instruct",
        num_games = 10
    )

    
        