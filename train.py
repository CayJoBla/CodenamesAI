import os
import numpy as np
from enum import Enum
import argparse
import json
import time

from codenames.gym import CodenamesEnv
from codenames.agents import Human, Dummy, Spymaster, Operative
from codenames import constants as cn

DEFAULT_AI_MODEL = "gpt2"
KEYWORDS = ['spymaster', 'operative', 'assassin', 'neutral', 'red', 'blue', 'hint', 'guess', 'codenames', 'word', 'board', 'number', 'team', 'game']

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
    ACTION_LENGTH = -0.5
    HINT_KEYWORD = -3

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
    device=None,
):
    # Initialize the agents
    team = cn.Team.RED
    if spymaster is None:
        spymaster = Human(team, cn.Role.SPYMASTER) if allow_human_player else Dummy(team, cn.Role.SPYMASTER)
    elif isinstance(spymaster, str):
        spymaster = Spymaster(team, model_name_or_path=spymaster, device=device)
    elif not isinstance(spymaster, Spymaster):
        raise ValueError("Invalid spymaster argument.")
    
    if operative is None:
        operative = Human(team, cn.Role.OPERATIVE) if allow_human_player else Dummy(team, cn.Role.OPERATIVE) 
    elif isinstance(operative, str):
        operative = Operative(team, model_name_or_path=operative, device=device)
    elif not isinstance(spymaster, Operative):
        raise ValueError("Invalid operative argument.")

    # Initialize the Codenames game environment
    env = CodenamesEnv(
        board_shape=board_shape,
        num_red=num_red, 
        num_blue=num_blue,
        num_assassin=num_assassin,
        word_file=word_file,
        reward_values=TrainReward
    )

    # Run the Codenames game
    spymaster_model_name = "human" if isinstance(spymaster, Human) else spymaster.ppo_trainer.config.model_name
    operative_model_name = "human" if isinstance(operative, Human) else operative.ppo_trainer.config.model_name
    training_log = {
        "spymaster_model": spymaster_model_name,
        "operative_model": operative_model_name,
        "board_shape": board_shape,
        "num_red": num_red,
        "num_blue": num_blue,
        "num_assassin": num_assassin,
        "word_file": word_file,
        "do_train": do_train,
        "num_games": num_games,
        "device": device,
        "games": []
    }
    for i in range(num_games):
        print(f"---------------- Game {i + 1} ----------------")
        env.reset()
        env.render()
        done = False
        start_time = time.time()
        game = {
            "game": i + 1,
            "words": env.state["words"].tolist(),
            "colors": [color.name for color in env.state["colors"]],
            "play": []
        }
        while not done:
            # Get team
            team = env.state['turn'][0]
            spymaster.team = team
            operative.team = team
            print(f"{team.name} Team's turn:\n")

            turn_desc = {
                "team": team.name,
                "spymaster": {},
                "operative": [],
            }

            # Spymaster's turn
            assert env.state['turn'] == (team, cn.Role.SPYMASTER)
            action = spymaster.get_action(env.state)
            print("Spymaster's Action:", action)
            turn_desc["spymaster"]["action"] = action
            _, reward_n, done, info = env.step(action)
            if do_train:
                action_words = action.lower().split()
                spymaster_reward = reward_n[(team, cn.Role.SPYMASTER)]
                if np.any(np.isin(KEYWORDS, action_words)):
                    spymaster_reward = min(spymaster_reward, TrainReward.HINT_KEYWORD.value)
                    info['result'] += " Penalty because a keyword was used."
                spymaster_reward += len(action_words) * TrainReward.ACTION_LENGTH.value
                print("Spymaster Initial Reward:", spymaster_reward)
                turn_desc["spymaster"]["initial_reward"] = spymaster_reward
                turn_desc["spymaster"]["info"] = info['result']
            print(f"Result: {info['result']}\n")

            # Operative's turn
            while not done and team == env.state['turn'][0]:
                assert env.state['turn'] == (team, cn.Role.OPERATIVE)
                turn_desc["operative"].append({})
                obs_state = {k: v for k, v in env.state.items() if k != 'colors'}
                action = operative.get_action(obs_state)
                print("Operative Action:", action)
                turn_desc["operative"][-1]["action"] = action
                _, reward_n, done, info = env.step(action)
                if do_train:
                    operative_reward = reward_n[(team, cn.Role.OPERATIVE)]
                    spymaster_reward += reward_n[(team, cn.Role.SPYMASTER)]
                    print("Operative Reward:", operative_reward)
                    turn_desc["operative"][-1]["reward"] = operative_reward
                    operative.step(operative_reward)
                print(f"Result: {info['result']}\n")
                turn_desc["operative"][-1]["info"] = info['result']

            if do_train:
                print(f"Spymaster Final Reward: {spymaster_reward}\n")
                turn_desc["spymaster"]["final_reward"] = spymaster_reward
                spymaster.step(spymaster_reward)

            game["play"].append(turn_desc)

        print(f"Game {i + 1} Over!")
        game["run_time"] = time.time() - start_time
        training_log["games"].append(game)

    if save_dir is not None:
        spymaster.save(os.path.join(save_dir, 
            f"{spymaster_model_name.split('/')[-1]}-{spymaster.role.name.lower()}"
        ))
        operative.save(os.path.join(save_dir, 
            f"{operative_model_name.split('/')[-1]}-{operative.role.name.lower()}"
        ))
        with open(os.path.join(save_dir, f"training_log.json"), "w") as f:
            json.dump(training_log, f)
        

    return spymaster, operative


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='SpymasterTrainer',
        description="Train a Codenames spymaster model to respond with valid hints."
    )
    parser.add_argument(
        "--spymaster", 
        type=str, 
        default=None, 
        help="Path to a spymaster model to use."
    )
    parser.add_argument(
        "--operative", 
        type=str, 
        default=None, 
        help="Path to an operative model to use."
    )
    parser.add_argument(
        "--board_shape", 
        type=int, 
        nargs=2, 
        default=(5, 5), 
        help="Shape of the Codenames board."
    )
    parser.add_argument(
        "--num_red", 
        type=int, 
        default=9, 
        help="Number of red words on the board."
    )
    parser.add_argument(
        "--num_blue", 
        type=int, 
        default=8, 
        help="Number of blue words on the board."
    )
    parser.add_argument(
        "--num_assassin", 
        type=int, 
        default=1, 
        help="Number of assassin words on the board."
    )
    parser.add_argument(
        "--word_file", 
        type=str, 
        default='codenames/word_lists/codenames.txt', 
        help="Path to the word list file."
    )
    parser.add_argument(
        "--allow_human_player", 
        action='store_true', 
        help="Allow a human player to play in the loop."
    )
    parser.add_argument(
        "--do_train", 
        action='store_true', 
        help="Train the spymaster and operative models."
    )
    parser.add_argument(
        "--num_games", 
        type=int, 
        default=1, 
        help="Number of games to play."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default=None, 
        help="Directory to save the trained models."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="Device to use for training."
    )
    args = parser.parse_args()
    codenames(**vars(args))

    
        