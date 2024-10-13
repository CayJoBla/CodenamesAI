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
        ppo_config=None,
        generation_kwargs=None,
        freeze_base_model_weights=True,
        device=None,
    ):
        super().__init__(team, role)
        device = device or "cuda" if torch.cuda.is_available() else "cpu"
        is_cpu = device == "cpu"
        ppo_config = ppo_config or PPOConfig(
            model_name = model_name_or_path,
            query_dataset = None,
            batch_size = 1,
            mini_batch_size = 1,
            learning_rate = 1e-5,
            accelerator_kwargs = {"cpu": is_cpu,},
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name_or_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.use_default_system_prompt = False

        if freeze_base_model_weights:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.v_head.parameters():
                param.requires_grad = True
        
        self.ppo_trainer = PPOTrainer(
            config = ppo_config,
            model = model,
            tokenizer = tokenizer,
        )
        self.generation_kwargs = generation_kwargs or {
            "min_length": 5,        # <start header> assistant <end header> RESPONSE <eot>
            "max_new_tokens": 15,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        self.query_tensor = None
        self.response_tensor = None

    def tokenize(self, sample):
        tokenized = self.ppo_trainer.tokenizer(sample, return_tensors='pt')
        return tokenized['input_ids'].squeeze().to(self.ppo_trainer.current_device)
    
    def postprocess(self, response):
        response = response.replace("<|start_header_id|>assistant<|end_header_id|>", "")
        response = response.replace("<|eot_id|>", "")
        # response = response.replace("\n", " ")
        return response.strip()

    def get_action(self, obs_state):
        prompt = self.get_prompt(obs_state)
        self.query_tensor= self.tokenize(prompt)
        self.response_tensor = self.ppo_trainer.generate(
            self.query_tensor, 
            return_prompt=False,
            **self.generation_kwargs
        ).squeeze()
        response = self.ppo_trainer.tokenizer.decode(self.response_tensor)
        return self.parse_action(self.postprocess(response))

    def step(self, reward):
        if self.query_tensor is None or self.response_tensor is None:
            raise ValueError("No query/response tensors to update.")
        stats = self.ppo_trainer.step(
            [self.query_tensor], 
            [self.response_tensor], 
            [torch.tensor(reward, dtype=float).to(self.ppo_trainer.current_device)]
        )
        self.query_tensor = None
        self.response_tensor = None

    def save(self, save_dir):
        self.ppo_trainer.model.save_pretrained(save_dir)

class Spymaster(AI):
    def __init__(
        self, 
        team,
        model_name_or_path, 
        ppo_config=None,
        generation_kwargs=None,
        freeze_base_model_weights=True,
        device=None,
    ):
        super().__init__(
            team, 
            cn.Role.SPYMASTER,
            model_name_or_path,
            ppo_config,
            generation_kwargs,
            freeze_base_model_weights,
            device,
        )
    
    def get_prompt(self, obs_state):
        # Build the context for the chat
        context = ""
        for color in cn.Color:
            context += f"{color.name} Words:\n"
            for i, word_color in enumerate(obs_state['colors']):
                if color == word_color:
                    word = obs_state['words'][i]
                    guessed = obs_state['guessed'][i]
                    context += f"{word}"
                    if guessed:
                        context += "\t(Already guessed)"
                    context += "\n"
            context += "\n"
        chat = [
            {
                "role": "system",
                "content": f"You are the playing the game Codenames as the {self.team.name} team's Spymaster.\nYou are only allowed to give a single word hint followed by a number of words that correspond to the hint.\nProvide no context or additional information.\nDo not use a hint word that is already on the board.\nProvide your hint word and number separated by a space in the format \"HINT NUMBER\".\n",
            }, 
            {
                "role": "context",
                "content": context,
            }, 
            {
                "role": "user", 
                "content": f"Please provide a single word hint and a number of related words to the operatives on your team such that they can guess some of the {self.team.name} Team's words without guessing any of the other team's words, neutral words, or the assassin word(s).",
            },
        ]
        return self.ppo_trainer.tokenizer.apply_chat_template(chat, tokenize=False)
    
    def parse_action(self, response):
        # matches = re.findall(r"[a-zA-Z ]+ \d+", response)
        # return " ".join(matches) if len(matches) > 0 else ""
        return response

class Operative(AI):
    def __init__(
        self, 
        team,
        model_name_or_path, 
        ppo_config=None,
        generation_kwargs=None,
        freeze_base_model_weights=True,
        device=None,
    ):
        super().__init__(
            team, 
            cn.Role.OPERATIVE,
            model_name_or_path,
            ppo_config,
            generation_kwargs,
            freeze_base_model_weights,
            device,
        )
    
    def get_prompt(self, obs_state):
        # Build the context for the chat
        context = f"Words on the board:\n"
        for word in obs_state['words'][~obs_state['guessed']]:
            context += str(word) + "\n"
        chat = [
            {
                "role": "system",
                "content": f"You are the playing the game Codenames as the {self.team.name} team's Operative.\nYour goal is to guess the word on the board that is related to the given hint.\nProvide no context or additional information.\nYour guess must be a word that is on the board.\nIf you are unsure, you may skip your turn by saying nothing (i.e. "").\n",
            }, 
            {
                "role": "context",
                "content": context,
            }, 
            {
                "role": "user", 
                "content": f"Please provide a guess word from the board that relates to the given hint: \"{obs_state['hint']}\".",
            },
        ]
        return self.ppo_trainer.tokenizer.apply_chat_template(chat, tokenize=False)
    
    def parse_action(self, response):
        # matches = re.findall(r"[a-zA-Z ]+", response)
        # return " ".join(matches) if len(matches) > 0 else ""
        return response
        