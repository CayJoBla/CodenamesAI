from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
import re
import os

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
        device=None,
    ):
        super().__init__(team, role)
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        ppo_config = ppo_config or PPOConfig(
            model_name = model_name_or_path,
            query_dataset = None,
            batch_size = 1,
            mini_batch_size = 1,
            learning_rate = 1e-6,
            accelerator_kwargs = {"device_placement": False},
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name_or_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.use_default_system_prompt = False
        
        self.ppo_trainer = PPOTrainer(
            config = ppo_config,
            model = model,
            tokenizer = tokenizer,
        )
        self.ppo_trainer.current_device = self.device
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
        return tokenized['input_ids'].squeeze()
    
    def postprocess(self, response):
        response = response.replace("<|start_header_id|>assistant<|end_header_id|>", "")
        response = response.replace("<|eot_id|>", "")
        # response = response.replace("\n", " ")
        return response.strip()

    def get_action(self, obs_state):
        prompt = self.get_prompt(obs_state)
        self.query_tensor = self.tokenize(prompt).to(self.device)
        self.response_tensor = self.ppo_trainer.generate(
            self.query_tensor, 
            return_prompt=False,
            **self.generation_kwargs
        ).squeeze().to(self.device)
        response = self.ppo_trainer.tokenizer.decode(self.response_tensor)
        return self.parse_action(self.postprocess(response))

    def step(self, reward):
        if self.query_tensor is None or self.response_tensor is None:
            raise ValueError("No query/response tensors to update.")
        stats = self.ppo_trainer.step(
            [self.query_tensor], 
            [self.response_tensor], 
            [torch.tensor(reward, dtype=float).to(self.device)]
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
        device=None,
    ):
        super().__init__(
            team=team, 
            role=cn.Role.SPYMASTER,
            model_name_or_path=model_name_or_path,
            ppo_config=ppo_config,
            generation_kwargs=generation_kwargs,
            device=device,
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
        return response

class Operative(AI):
    def __init__(
        self, 
        team,
        model_name_or_path, 
        ppo_config=None,
        generation_kwargs=None,
        device=None,
    ):
        super().__init__(
            team, 
            cn.Role.OPERATIVE,
            model_name_or_path,
            ppo_config,
            generation_kwargs,
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

class AISharedAgent:
    def __init__(
        self,
        model_name_or_path,
        ppo_config=None,
        generation_kwargs=None,
        device=None,
    ):
        # Device Management
        device = device or "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and torch.cuda.device_count() > 1:
            self.llm_device = "cuda:0"
            self.spymaster_device = "cuda:1"
            self.operative_device = "cuda:1"
        else:
            self.llm_device = device
            self.spymaster_device = device
            self.operative_device = device

        # Load model and tokenizer
        self.model_name_or_path = model_name_or_path
        self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.llm_device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.use_default_system_prompt = False

        self.spymaster = AutoModelForCausalLMWithValueHead(self.llm)
        self.spymaster.v_head.to(self.spymaster_device)
        self.spymaster.is_peft_model = False
        spymaster_ref_model = AutoModelForCausalLMWithValueHead(self.llm)
        spymaster_ref_model.v_head.to(self.spymaster_device)
        spymaster_ref_model.is_peft_model = False
        
        self.operative = AutoModelForCausalLMWithValueHead(self.llm)
        self.operative.v_head.to(self.operative_device)
        self.operative.is_peft_model = False
        operative_ref_model = AutoModelForCausalLMWithValueHead(self.llm)
        operative_ref_model.v_head.to(self.operative_device)
        operative_ref_model.is_peft_model = False

        # Get configuration for the PPO Trainer
        ppo_config = ppo_config or PPOConfig(
            model_name = self.model_name_or_path,
            query_dataset = None,
            batch_size = 1,
            mini_batch_size = 1,
            learning_rate = 1e-5,
            accelerator_kwargs = {"device_placement": False},
        )
        
        # Freeze all layers except the language model head
        for name, param in self.llm.named_parameters():
            if "lm_head" not in name:
                param.requires_grad = False

        # Initialize the PPO Trainers
        self.ppo_trainer = {}
        self.ppo_trainer[cn.Role.SPYMASTER] = PPOTrainer(
            config = ppo_config,
            model = self.spymaster,
            ref_model = spymaster_ref_model,
            tokenizer = self.tokenizer,
        )
        self.ppo_trainer[cn.Role.OPERATIVE] = PPOTrainer(
            config = ppo_config,
            model = self.operative,
            ref_model = operative_ref_model,
            tokenizer = self.tokenizer,
        )
        self.ppo_trainer[cn.Role.SPYMASTER].current_device = self.llm_device
        self.ppo_trainer[cn.Role.OPERATIVE].current_device = self.llm_device

        # Set generation kwargs
        self.generation_kwargs = generation_kwargs or {
            "min_length": 5,        # <start header> assistant <end header> RESPONSE <eot>
            "max_new_tokens": 15,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        self.query_tensor = {cn.Role.SPYMASTER: None, cn.Role.OPERATIVE: None}
        self.response_tensor = {cn.Role.SPYMASTER: None, cn.Role.OPERATIVE: None}

    def tokenize(self, sample):
        tokenized = self.tokenizer(sample, return_tensors='pt')
        return tokenized['input_ids'].squeeze()
    
    def postprocess(self, response):
        response = response.replace("<|start_header_id|>assistant<|end_header_id|>", "")
        response = response.replace("<|eot_id|>", "")
        return response.strip()

    def get_prompt(self, obs_state):
        team, role = obs_state['turn']
        if role == cn.Role.SPYMASTER:
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
                    "content": f"You are the playing the game Codenames as the {team.name} team's Spymaster.\nYou are only allowed to give a single word hint followed by a number of words that correspond to the hint.\nProvide no context or additional information.\nDo not use a hint word that is already on the board.\nProvide your hint word and number separated by a space in the format \"HINT NUMBER\".\n",
                }, 
                {
                    "role": "context",
                    "content": context,
                }, 
                {
                    "role": "user", 
                    "content": f"Please provide a single word hint and a number of related words to the operatives on your team such that they can guess some of the {team.name} Team's words without guessing any of the other team's words, neutral words, or the assassin word(s).",
                },
            ]

        elif role == cn.Role.OPERATIVE:
            context = f"Words on the board:\n"
            for word in obs_state['words'][~obs_state['guessed']]:
                context += str(word) + "\n"
            chat = [
                {
                    "role": "system",
                    "content": f"You are the playing the game Codenames as the {team.name} team's Operative.\nYour goal is to guess the word on the board that is related to the given hint.\nProvide no context or additional information.\nYour guess must be a word that is on the board.\nIf you are unsure, you may skip your turn by saying nothing (i.e. "").\n",
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
        return self.tokenizer.apply_chat_template(chat, tokenize=False)

    def get_action(self, obs_state):
        role = obs_state['turn'][1]
        prompt = self.get_prompt(obs_state)
        self.query_tensor[role] = self.tokenize(prompt).to(self.llm_device)
        self.response_tensor[role] = self.ppo_trainer[role].generate(
            self.query_tensor[role], 
            return_prompt=False,
            **self.generation_kwargs
        ).squeeze().to(self.llm_device)
        response = self.tokenizer.decode(self.response_tensor[role])
        return self.postprocess(response)

    def step(self, reward, role):
        if self.query_tensor[role] is None or self.response_tensor[role] is None:
            raise ValueError("No query/response tensors to update.")
        stats = self.ppo_trainer[role].step(
            [self.query_tensor[role]], 
            [self.response_tensor[role]], 
            [torch.tensor(reward, dtype=float).to(self.llm_device)]
        )
        self.query_tensor[role] = None
        self.response_tensor[role] = None

    def save(self, save_dir):
        self.ppo_trainer[cn.Role.SPYMASTER].model.save_pretrained(
            os.path.join(save_dir, f"{self.model_name_or_path.split('/')[-1]}-Spymaster")
        )
        self.ppo_trainer[cn.Role.OPERATIVE].model.save_pretrained(
            os.path.join(save_dir, f"{self.model_name_or_path.split('/')[-1]}-Operative")
        )