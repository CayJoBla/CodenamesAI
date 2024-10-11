from codenames.agents import Spymaster
from codenames.constants import Team

spymaster = Spymaster(
    team=Team.RED,
    model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
)