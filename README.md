# Codenames SpyMaster

## Description

The purpose of this project is to create an agent that can play the game Codenames as the Spymaster. The agent will be able to generate clues based on the remaining words for their team and taking the into account the other words on the board to avoid. 

## Implementation

For this project, I decided to use CrewAI for LLM agents. I will be using the LLaMA 3 open-source model as the underlying model for each of the agents.

## Other ideas

- Incorporate the hints and guesses from the other team to take into account the thinking of the other team. (May be too complex for the model)
- Create a version to play as the field operative (guesser)