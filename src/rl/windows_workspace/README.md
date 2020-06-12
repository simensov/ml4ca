# windows workspace
This is the workspace in which all development, training and local testing of the RL algorithms are done, and is meant for running on a windows computer with the Cybersea simulator.

It builds on OpenAI's Spinning Up (https://spinningup.openai.com/en/latest/), but is different in that it is only compatible with tensorflow 1. If pytorch if your backend of choice, this is easily done going from Spinning Up's repository.

## How to use

### train.py
Launches simulator, loads configuration, and starts training of an RL agent, using the hyperparameters specified in the argparse. The hyperparameter choice is saved in config.json.

example:
    - python .\train.py --sim 0 --exp_name experimentname --note 'Allows an explanation of e.g. the reward function so that the information is stored in the config.json file'

This launches simulator 0, given a certain name and a certain note.

### plot_results.py
Plots the training progress of different experiments. 

example:
    - python .\plot_results.py '.\data\experiment\' --value AverageEpRet LossV AverageVVals Entriopy --smooth 25

This plots four time series of the four given logged parameters, and shows the running average of the 25 previous results. The default x axis is the number of steps performed in the simulator (environment interactions), but can be changed with --xaxis Param, where Param could be e.g. Time.


### load_and_test.py
This script loads a policy and tests it in a simulator with the VesselMainView as an option. If '--plot True' (which it is by default), it plots the trajectories and reward developments over time during testing. 

example:
    - python .\load_and_test.py '.\data\experiment\experiment_s0'

Note that it is important to give the lowest path to the script, as e.g. '.\data\experiment\' will not work (the directory has to contain a tf1_save directory)


### Tips and tricks
    - A known error is, both for train.py and load_and_test.py, when trying to launch a simulator and load its configuration (if the simulator hasn't already been opened or a userdir has not been previosly established) is: 
        - py4j.protocol.Py4JNetworkError: An error occurred while trying to connect to the Java server (127.0.0.1:25344)

        This is no big problem; the simulator windows opens, but the configuration is not loaded and the Python script terminates. Just load the configuration manually in the simulator window, and launch the Python script when completed. 

    - Another known error is that using too long experiment names makes the storage of variables crash somehow. Annoyingly, this happens after the simulation starts seemingly without problems, but the error occurs during the first attempt to save variables and weights. The printout is something along the lines of that it cannot find a certain operation, referring to a certain weight in the network. Just keep the names short, lowercase letters, avoid numbers and symbols, and you will be fine.


## Other directories

### specific
This directory contains the implementations which launches the simulator, interacts with the simulator from the created ReVolt-environment etc.

### spinup
This directory contains the PPO implmentation which comes from Spinning Up. The code has also been made compatible with the TRPO algorithm, although it was not used in the thesis.