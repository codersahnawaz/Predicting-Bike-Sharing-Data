Observation:-
=============
1. I have noticed if we don't the transpose in loss calculation, my model can reach to some optimal solution.

    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    
2. Also while ploting data vs prediction, if we don't take the transpose, i can see prediction. With transpose, i dont get to see the
   prediction at all.

    predictions = network.run(test_features).T*std + mean
    
so my tentative solution is without taking the transpose in above 2 cases. (please advise)

3. with transpose, validation loss stays around 0.9-1.4 range with all kind of different combinations of hyperparameters.
4. with transpose, it takes very long to train the model.

Please give your detailed insights on the above points. Thanks!

===========================================================================================================================================
Below are some of the results i got while training with different parameters, finally picked the following as the most optimal among the results.

most optimal hyperparameters : -
-----------------------------
iterations = 5000
learning_rate = 0.4
hidden_nodes = 10
output_nodes = 1

===========================================================================================================================================

Below are some of the results how i got to the solution.

Pass 1
=========

iterations = 3000
learning_rate = 0.1
hidden_nodes = 5
output_nodes = 1

Progress: 100.0% ... Training loss: 0.259 ... Validation loss: 0.434

Pass 2
============

iterations = 500
learning_rate = 0.1
hidden_nodes = 10
output_nodes = 1

Progress: 99.8% ... Training loss: 0.412 ... Validation loss: 0.665

Pass 3
============

iterations = 1000
learning_rate = 0.1
hidden_nodes = 10
output_nodes = 1

Progress: 99.9% ... Training loss: 0.310 ... Validation loss: 0.489

Pass 4
============

iterations = 1500
learning_rate = 0.1
hidden_nodes = 10
output_nodes = 1

Progress: 99.9% ... Training loss: 0.285 ... Validation loss: 0.455

Pass 5
============

iterations = 5000
learning_rate = 0.1
hidden_nodes = 10
output_nodes = 1

Progress: 100.0% ... Training loss: 0.238 ... Validation loss: 0.409

Pass 6
============

iterations = 5000
learning_rate = 0.2
hidden_nodes = 15
output_nodes = 1

Progress: 100.0% ... Training loss: 0.146 ... Validation loss: 0.292

Pass 7
============

iterations = 5000
learning_rate = 0.4
hidden_nodes = 15
output_nodes = 1

Progress: 100.0% ... Training loss: 0.071 ... Validation loss: 0.162

Pass 8
============

iterations = 5000
learning_rate = 0.4
hidden_nodes = 10
output_nodes = 1

Progress: 100.0% ... Training loss: 0.064 ... Validation loss: 0.156

Pass 9
============

iterations = 2500
learning_rate = 0.4
hidden_nodes = 10
output_nodes = 1

Progress: 99.8% ... Training loss: 0.062 ... Validation loss: 0.155

Pass 10
============

iterations = 2500
learning_rate = 0.4
hidden_nodes = 10
output_nodes = 1

Progress: 99.8% ... Training loss: 0.075 ... Validation loss: 0.255