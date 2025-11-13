# sturdy-carnival
Testing whether the accepted values of chess pieces are accurate by fitting known distributions to the dataset made, and using a mixture density network (MDN).

NOTE: 
To run this code you will need to install stockfish (or another engine), then replace the path in the code with your own in 
worker.py

worker.py:
Responsible for running stockfish to create random positions,
then getting the evaluations before and after a piece is removed from the board to estimate the value of the piece

controller.py:
Runs worker.py in batches and ensures there are no threading and stockfish errors (stockfish has incompatibility issues with windows). Saves the piece values of each piece after each position in a .pkl file after combining batches

combine_pieces.py:
Combines all the piece values for each position into 1 combined_piece_values.pkl file

values_estim.py:
Uses a maxwell-boltzmann and gamma distributions to fit the data, the base values (e.g. Rook = 5) for each piece usually lie in the middle of the peaks (most probable) values of each distribution.
Overall it's not a terrible fit but most likely underestimates the value of the queen and rook.

train_MDN.py:
Uses a mixture density network (MDN) to estimate the probability distribution of a piece having a specific value instead of trying to fit a known distribution. Uses 3 gaussian distributions to fit to the data. Has 64 total hidden neurons and 2 hidden layers. 3 output layers a different one that corresponds to the weighting of each gaussian, its mean, and standard deviation. The results are saved in a .pth file for each piece. The model might be overfitted to each pieces data.

values_MDN.py:
Reads the network weights and outputs the parameters for each gaussian for each piece.

/piece_values:
Stores the .pkl files for each piece from controller.py

/piece_weights_nn:
Stores the MDN weights from train_MDN for each piece

/figs:
stores the outputted plots, files ending in _est.png are from values_estim.py where we fit known distributions to the data. Files ending in _MDN.png is the distribution formed from the MDN, and the files ending in _contributions.png are box plots to see how each gaussian contributes to the final fit for the MDN.