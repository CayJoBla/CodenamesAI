import os

from .gym import CodenamesEnv
from .agents import Human, Dummy, Spymaster, Operative
from . import constants as cn

DEFAULT_AI_MODEL = "gpt2"

def codenames(
    agents={},
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
    )

    # Initialize the agents
    agent_name_map = {
        'spymaster_red': (cn.Team.RED, cn.Role.SPYMASTER),
        'operative_red': (cn.Team.RED, cn.Role.OPERATIVE),
        'spymaster_blue': (cn.Team.BLUE, cn.Role.SPYMASTER),
        'operative_blue': (cn.Team.BLUE, cn.Role.OPERATIVE),
    }
    if 'spymaster' in agents:
        agents['spymaster_red'] = agents['spymaster']
        agents['spymaster_blue'] = agents['spymaster']
        agents.pop('spymaster')
    if 'operative' in agents:
        agents['operative_red'] = agents['operative']
        agents['operative_blue'] = agents['operative']
        agents.pop('operative')
    for k, v in agent_name_map.items():
        agents[v] = agents.pop(k, None)

    # Load agent models
    for k, agent in agents.items():
        team, role = k
        if agent is None or agent.strip().lower() == 'dummy':
            agents[k] = Dummy(team, role)
            continue
        elif agent == 'human':
            if not allow_human_player:
                raise ValueError("Please set allow_human_player=True to allow human players.")
            agents[k] = Human(team, role)
            continue
        elif agent == "ai" or agent == "computer":
            agent = DEFAULT_AI_MODEL

        if role == cn.Role.SPYMASTER:
            agents[k] = Spymaster(team, model_name_or_path=agent)
        elif role == cn.Role.OPERATIVE:
            agents[k] = Operative(team, model_name_or_path=agent)
        else:
            raise ValueError(f"Invalid role for agent: {role}")

    # Run the Codenames game
    for game in range(num_games):
        env.reset()
        done = False
        while not done:
            # Get team
            team = env.state['turn'][0]
            spymaster = agents[(team, cn.Role.SPYMASTER)]
            operative = agents[(team, cn.Role.OPERATIVE)]

            # Spymaster's turn
            assert env.state['turn'][1] == cn.Role.SPYMASTER
            action = spymaster.get_action(env.state)
            obs, reward_n, done, info = env.step(action)
            spymaster_reward = reward_n[(team, cn.Role.SPYMASTER)]
            print("Spymaster's Action:", action)
            print("Current Reward:", spymaster_reward)
            print("Info:", info)

            # Operative's turn
            while not done and team == env.state['turn'][0]:
                assert env.state['turn'][1] == cn.Role.OPERATIVE
                obs_state = {k: v for k, v in env.state.items() if k != 'board'}
                action = operative.get_action(obs_state)
                obs, reward_n, done, info = env.step(action)
                if not operative.is_dummy:
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
        for k, agent in agents.items():
            team, role = k
            agent_dir = os.path.join(save_dir, f"{team.name}_{role.name}")
            agent.save(agent_dir)

    return agents


if __name__ == "__main__":
    codenames(
        agents = {
            "spymaster": "meta-llama/Llama-3.2-1B-Instruct",
            "operative": "dummy",
        },
        num_games = 10
    )

    
        