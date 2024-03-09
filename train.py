# train.py is the main file to train the model
#for one game:
initialize model 
initialize opponent (either itself, other chess bot, past version of itself?)

initialize blank chess board

coin flip decide black/white players

inputs to model look like: (bit for black or white, known board, turn count, opponent's seen board)
                            
initiliaze opponent's seen board (piece, location, # turn seen, bit for captured or not)

while both kings are present on the board: