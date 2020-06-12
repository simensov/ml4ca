# ml4ca :speedboat: :space_invader:
## Machine Learning for Control Allocation

This work comprise the project thesis, TMR4510, and masters thesis, TMR4930, at NTNU, fall 2019/spring 2020. It investigates different machine learning methods for finding control inputs to the azimuth thrusters of the ReVolt model ship from DNVGL. Each of the directories in src/ contains their own README files which explains how the code was used during the thesis, and how to run it yourself. The only requirement for using the code directly is access to the Cybersea simulator, which only DVNGL can grant access to. But the code is easily adjustable for any other simulator, and in src/rl/windows_workspace, you can find examples of how an environment was created of the simulator by using Python.

## Reinforcement Learning (src/rl/):
Masters thesis: Going from the project thesis, it was expected that RL could improve the results from supervised learning due to difficulties of problem formulation using the latter. The PPO algorithm from OpenAI was chosen as RL algorithm, and the implementation is based on the work from the Spinning Up repository, also from OpenAI. The outcome resulted in a control scheme directly translating desired pose of the vessel into thruster commands, hence replacing both the traditional motion controller (like a PID or an LQR) and the thrust allocation scheme (e.g. pseudoinverse or quadratic programming). The performance was tested against these classic methods, and was found to be as good or better than the others in terms of station-keeping accuracy while simultaneously being more energy efficient.

The implementation was built on OpenAI's Spinning Up repository (https://spinningup.openai.com/), and the work itself is summarized in the master's thesis "Energy Efficient Dynamic Positioning of Marine Surface Vessels using Deep Reinforcement Learning", submitted in June 2020. All the work has been made public for clarity, and for ease of use for others looking to work with Deep Reinforcement Learning for controlling vehicles.

## Supervised Learning (src/supervised/):
Project thesis: A dataset was created, used for training a neural network using supervised learning in order to learn the physics between the forces/moments acting on ReVolt model ship and the commanded propeller RPMs and angles. A quadratic programming approach was also established as a benchmark (in src/ql/). The SL version resulted in a satisfactory thrust allocation scheme, but the quadratic programming approach was superior in terms of fuel consumption and more controlled motion of the vessel.

## Use a virtual environment to adjust package versions

The supervised learning and reinforcement learning implementations uses different versions of e.g. tensorflow and keras. This was due to that the specific CPU on the computer used was optimized with tensorflow 1.10.0 and keras 2.2.0, used for training the supervised learning model with large amounts of data. But since the PPO algorithm was trained online, using another computer, using Windows (the same one running the ReVolt simulator), updated versions of the packages was used for better documentation and compatibility.

All in all, a virtual environment avoids ruining the other projects on your machine that might depend on other versions of packages used in this project.

- Installation: https://virtualenv.pypa.io/en/latest/installation.html
- Create a virtual environment within you project folder: `virtualenv --system-site-packages -p python3 ./venv`
- To activate: `source venv/bin/activate` -> (venv) should pop up to the left on the command line.
- Use `which python` to confirm that the shell uses the python inside venv/. Use `pip freeze` to see the current package versions used in the virtual environment.
- To deactivate: `deactivate`
