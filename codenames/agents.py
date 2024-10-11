from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
import re

from . import constants as cn


class CodenamesAgent:
    def __init__(self, team, role):
        self.team = team
        self.role = role
        self.is_human = False
        self.is_dummy = False

    def __repr__(self):
        return f"{self.__class__.__name__}({self.team.name},{self.role.name})"
    
class Dummy(CodenamesAgent):
    def __init__(self, team, role):
        super().__init__(team, role)
        self.is_dummy = True

    def get_action(self, observation_state):
        return ""
    
    def step(self, reward):
        pass

class Human(CodenamesAgent):
    def __init__(self, team, role):
        super().__init__(team, role)
        self.is_human = True

    def get_action(self, observation_state):
        print(f"You are the {self.team.name} {self.role.name}.", flush=True)
        print("Observation State:", observation_state, flush=True)
        action = input("Enter your action: ")
        return action
    
    def step(self, reward):
        pass

class AI(CodenamesAgent):
    def __init__(
        self,
        team,
        role, 
        model_name_or_path,
        device=None,
        **config_kwargs
    ):
        super().__init__(team, role)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name_or_path).to(self.device)

        self.ppo_config = PPOConfig(
            model_name = model_name_or_path,
            batch_size = 1,
            mini_batch_size = 1,
            **config_kwargs
        )
        self.ppo_trainer = PPOTrainer(
            config = self.ppo_config,
            model = model,
            tokenizer = self.tokenizer,
        )
        self.generation_kwargs = {
            "min_length": 5,
            "max_new_tokens": 50,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        self.query_tensor = None
        self.response_tensor = None

    def tokenize(self, sample):
        tokenized = self.tokenizer(sample, return_tensors='pt')
        return tokenized['input_ids'].squeeze().to(self.device)

    def get_action(self, obs_state):
        prompt = self.get_prompt(obs_state)
        self.query_tensor= self.tokenize(prompt)
        self.response_tensor = self.ppo_trainer.generate(
            self.query_tensor, 
            return_prompt=False,
            **self.generation_kwargs
        ).squeeze()
        response = self.tokenizer.decode(self.response_tensor)
        return self.parse_action(response)

    def step(self, reward):
        if self.query_tensor is None or self.response_tensor is None:
            raise ValueError("No query/response tensors to update.")
        stats = self.ppo_trainer.step(
            [self.query_tensor], 
            [self.response_tensor], 
            [torch.tensor(reward, dtype=float).to(self.device)]
        )
        # self.ppo_trainer.log_stats(stats, None, reward)
        self.query_tensor = None
        self.response_tensor = None

    def save_model(self, path):
        self.ppo_trainer.save_model(path)


class Spymaster(AI):
    def __init__(
        self, 
        team,
        model_name_or_path, 
        device=None,
        **config_kwargs
    ):
        super().__init__(
            team, 
            cn.Role.SPYMASTER,
            model_name_or_path,
            device,
            **config_kwargs
        )
        self.tokenizer.use_default_system_prompt = False

    def __repr__(self):
        return f"{self.__class__.__name__}({self.team.name})"
    
    def get_prompt(self, obs_state):
        print("Observation State:", obs_state)
        chat = [
            {
                "role": "system",
                "content": f"You are the playing the game Codenames as the {self.team.name} team's Spymaster.\nYou are only allowed to give a single word hint followed by a number of words that correspond to the hint.\nProvide no context or additional information.\nDo not use a hint word that is already on the board.\nProvide your hint word and number separated by a space in the format \"HINT NUMBER\".\n",
            }, 
            {
                "role": "context",
                "content": "",
            }, 
            {
                "role": "user", 
                "content": f"Please provide a single word hint and a number of related words to the operatives on your team such that they can guess some of the {self.team.name} Team's words without guessing any of the other team's words, neutral words, or the assassin word(s).",
            },
        ]
        context = ""
        for color in cn.Color:
            context += f"{color.name} Words:\n"
            for i, word_color in enumerate(obs_state['board']):
                if color == word_color:
                    word = obs_state['words'][i]
                    guessed = obs_state['guessed'][i]
                    context += f"{word}"
                    if guessed:
                        context += "\t(Already guessed)"
                    context += "\n"
            context += "\n"
        chat[1]["content"] = context
        
        return self.tokenizer.apply_chat_template(chat, tokenize=False)
    
    def parse_action(self, response):
        print("Response:", response)
        matches = re.findall(r"[a-zA-Z ]+ \d+", response)
        if len(matches) == 0:
            return ""
        elif len(matches) >= 1:     # Only return the first hint
            return matches[0]


class Operative(AI):
    def __init__(
        self,
        team,
        model_name_or_path,
        device=None,
        **config_kwargs
    ):
        super().__init__(
            team, 
            cn.Role.OPERATIVE,
            model_name_or_path,
            device,
            **config_kwargs
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.team.name})"
    
    def get_prompt(self, obs_state):
        prompt = f"You are the playing the game Codenames.\n"
        prompt += f"Your role is the {self.team.name} team's {self.role.name}.\n"
        prompt += f"Please provide a guess word from the board that corresponds to the Spymaster's hint.\n"
        prompt += f"Words on the board:\n"
        for word, guessed in zip(obs_state['words'], obs_state['guessed']):
            prompt += str(word)
            if guessed:
                prompt += "\t(Already guessed)"
            prompt += "\n"
        prompt += "\n"
        prompt += f"Spymaster's Hint: {obs_state["hint"]}\n"
        prompt += "Your Guess: "
        return prompt
        