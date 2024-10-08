from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch

from . import constants as cn


class CodenamesAgent:
    def __init__(self, role):
        self.role = role

    def __repr__(self):
        return f"{self.__class__.__name__}({self.role})"

    def get_action(self, obs_state):
        return ""

    def step(self, reward):
        pass

class UserAgent(CodenamesAgent):
    def get_action(self, observation_state):
        print(f"You are the {self.role}.", flush=True)
        print("Observation State:", observation_state, flush=True)
        action = input("Enter your action: ")
        return action


class Spymaster(CodenamesAgent):
    def __init__(
        self, 
        team,
        model_name_or_path, 
        device=None,
        **config_kwargs
    ):
        super().__init__((team, cn.Role.SPYMASTER))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ppo_config = PPOConfig(
            model_name = model_name_or_path,
            **config_kwargs
        )
        self.ppo_trainer = PPOTrainer(
            config = self.ppo_config,
            model = self.model,
            tokenizer = self.tokenizer,
        )
        self.generation_kwargs = {
            "min_length": 0,
            "max_new_tokens": 50,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        self.query_tensors = None
        self.response_tensors = None

    def tokenize(self, sample):
        tokenized = self.tokenizer(sample, return_tensors='pt')
        return tokenized['input_ids'].squeeze().to(self.device)
    
    def get_prompt(self, obs_state):
        team, role = self.role
        prompt = f"You are the playing the game Codenames.\n"
        prompt += f"Your role is the {team.name} team's {role.name}.\n"
        prompt += f"Please provide a hint to the operatives on your team such that they guess the {team.name} Team's words without guessing any of the other team's words, neutral words, or the assassin word(s).\n"
        prompt += f"Do not use a hint word that is already on the board.\n"
        prompt += f"Provide your hint in the format \"HINT COUNT\".\n\n"
        prompt += f"Board State:\n"
        for color in cn.Color:
            prompt += f"{color.name} Words:\n"
            for i, word_color in enumerate(obs_state['board']):
                if color == word_color:
                    word = obs_state['words'][i]
                    guessed = obs_state['guessed'][i]
                    prompt += f"{word}"
                    if guessed:
                        prompt += "\t(Already guessed)"
                    prompt += "\n"
            prompt += "\n"

        prompt += "Your Hint: "

        return prompt
            

    def parse_response(self, response):
        return response

    def get_action(self, obs_state):
        prompt = self.get_prompt(obs_state)
        self.query_tensors= self.tokenize(prompt)
        self.response_tensors = self.ppo_trainer.generate(
            self.query_tensors, 
            return_prompt=False,
            **self.generation_kwargs
        )
        response = self.tokenizer.decode(self.response_tensors.squeeze())
        action = self.parse_response(response)
        return action

    def step(self, reward):
        if self.query_tensors is None or self.response_tensors is None:
            raise ValueError("No query/response tensors to update.")
        stats = self.ppo_trainer.step(
            self.query_tensors, 
            self.response_tensors, 
            reward
        )
        self.ppo_trainer.log_stats(stats, None, reward)
        self.query_tensors = None
        self.response_tensors = None

    def save_model(self, path):
        self.ppo_trainer.save_model(path)

# TODO: Actually code this correctly
class Operative(CodenamesAgent):
    def __init__(
        self,
        team,
        model_name_or_path,
        device=None,
        **config_kwargs
    ):
        super().__init__((team, cn.Role.OPERATIVE))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name_or_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ppo_config = PPOConfig(
            model_name = model_name_or_path,
            **config_kwargs
        )
        self.ppo_trainer = PPOTrainer(
            config = self.ppo_config,
            model = self.model,
            tokenizer = self.tokenizer,
        )
        self.generation_kwargs = {
            "min_length": 0,
            "max_new_tokens": 50,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        self.query_tensors = None
        self.response_tensors = None
        