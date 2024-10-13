import os
import numpy as np
import argparse

from trl import PPOConfig

from codenames.gym import CodenamesEnv
from codenames.agents import Human, Spymaster
from codenames import constants as cn

DEFAULT_AI_MODEL = "gpt2"
KEYWORDS = ['spymaster', 'operative', 'assassin', 'neutral', 'red', 'blue', 'hint', 'guess', 'codenames', 'word', 'board', 'number', 'team', 'game']
WORD_REWARD = -0.5                                  # Reward for each word in the action to encourage shorter hints
KEYWORD_REWARD = cn.Reward.HINT_WORD_ON_BOARD.value # Max reward for action that contains a keyword

def codenames_spymaster(
    spymaster,
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
    # Initialize the spymaster agent
    team = cn.Team.RED
    if spymaster is None:
        if allow_human_player:
            spymaster = Human(team, cn.Role.SPYMASTER)
        else:
            spymaster = Spymaster(team, model_name_or_path=DEFAULT_AI_MODEL, device=device)
    elif isinstance(spymaster, str):
        spymaster = Spymaster(team, model_name_or_path=spymaster, device=device)
    elif not isinstance(spymaster, Spymaster):
        raise ValueError("Invalid spymaster argument.")

    # Initialize the Codenames game environment
    env = CodenamesEnv(
        board_shape=board_shape,
        num_red=num_red, 
        num_blue=num_blue,
        num_assassin=num_assassin,
        word_file=word_file,
    )

    # Run the Codenames game
    step = 0
    while step < num_games:
        env.reset()
        print(env.state)
        for _ in range(2):
            # Do some clean substeps with no words guessed yet, then 
            # some substeps with words guessed already
            for team in [cn.Team.RED, cn.Team.BLUE]:
                spymaster.team = team
                if env.state['turn'][0] != team:
                    env.turnover()

                # Spymaster's turn
                assert env.state['turn'] == (team, cn.Role.SPYMASTER)
                action = spymaster.get_action(env.state)
                print("Spymaster's Action:", action)
                _, reward_n, _, info = env.step(action)
                if do_train:
                    action_words = action.lower().split()
                    spymaster_reward = reward_n[(team, cn.Role.SPYMASTER)]
                    if np.any(np.isin(KEYWORDS, action_words)):
                        spymaster_reward = min(spymaster_reward, KEYWORD_REWARD)
                        info['result'] += " Penalty because a keyword was used."
                    spymaster_reward += WORD_REWARD * len(action_words)
                    print("Reward:", spymaster_reward)
                    spymaster.step(spymaster_reward)
                print("Info:", info)

            # Randomly choose words as guessed already (after first substep per team)
            if num_red > 1 and num_blue > 1:
                indices = np.random.choice(np.arange(env.num_words), size=min(num_red,num_blue)-1, replace=False)
                env.state['guessed'][indices] = True
        step += 1

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        spymaster.save(save_dir)

    return spymaster


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='SpymasterTrainer',
        description="Train a Codenames spymaster model to respond with valid hints."
    )
    parser.add_argument(
        "--spymaster", 
        type=str, 
        default=None, 
        help="Path to a trained spymaster model."
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
        help="Allow a human player to play as the spymaster."
    )
    parser.add_argument(
        "--do_train", 
        action='store_true', 
        help="Train the spymaster model."
    )
    parser.add_argument(
        "--num_games", 
        type=int, 
        default=1, 
        help="Number of 'games' to play."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default=None, 
        help="Directory to save the trained spymaster model."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None, 
        help="Device to use for training."
    )
    args = parser.parse_args()
    codenames_spymaster(**vars(args))

    
        