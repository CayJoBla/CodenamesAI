import os

from .gym import CodenamesEnv
from .agents import Human, Dummy, Spymaster, Operative
from . import constants as cn

DEFAULT_AI_MODEL = "gpt2"

def codenames_spymaster(
    spymaster,
    board_shape=(5, 5),
    num_red=9,
    num_blue=8,
    num_assassin=1,
    word_file='codenames/word_lists/codenames.txt',
    allow_human_player=False,
    do_train=False,
    num_steps=1,
    save_dir=None,
):
    team = cn.Team.RED
    if spymaster is None and not allow_human_player:
        if allow_human_player:
            spymaster = Human(team, cn.Role.SPYMASTER)
        else:
            spymaster = Spymaster(team, model_name_or_path=DEFAULT_AI_MODEL)
    elif isinstance(spymaster, str):
        spymaster = Spymaster(team, model_name_or_path=spymaster)
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
    while step < num_steps:
        env.reset()
        for substep in range(2):
            # Do clean substeps with no words guessed yet, then substeps with 
            # words guessed already
            for team in [cn.Team.RED, cn.Team.BLUE]:
                spymaster.team = team
                if env.state['turn'][0] != team:
                    env.turnover()

                # Spymaster's turn
                assert env.state['turn'] == (team, cn.Role.SPYMASTER)
                action = spymaster.get_action(env.state)
                obs, reward_n, done, info = env.step(action)
                spymaster_reward = reward_n[(team, cn.Role.SPYMASTER)]
                print("Spymaster's Action:", action)
                print("Reward:", spymaster_reward)
                print("Info:", info)

                if do_train:
                    spymaster.step(spymaster_reward)

            # Randomly choose words as guessed already (after first substep per team)
            if num_red > 1 and num_blue > 1:
                indices = random.choice(range(env.num_words), min(num_red, num_blue)-1)
                for i in indices:
                    env.state['guessed'][i] = True
        step += 1

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        spymaster.save(os.path.join(save_dir, f"{role.name}"))

    return spymaster


if __name__ == "__main__":
    codenames(
        spymaster = "meta-llama/Llama-3.2-1B-Instruct",
        num_games = 10
    )

    
        