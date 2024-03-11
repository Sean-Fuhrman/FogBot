# FogBot

## Overall Strategy

We have 2 bots, One that predicts the value of board states. Our method is based on predicting the most valuable next-board position and choosing an action based on that. 


## TODOS:
1. Create Board wrapper
2. Create Train Loop

## How to run
1. Make sure config.yaml contains proper configs
2. Run training or playing:
   - For training: `python train.py`
   - For playing:  `python play.py`



## File structure

1. model.py - contains model
2. train.py - contains training loop for model
3. play.py - file to run to play against / visualize model play
4. window.py - contains code to run window
5. models/ - folder containing models generate
6. config.py - contains configurations for train.py and play.py