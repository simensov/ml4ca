# Implementation of actor critic using Proximal Policy Optimization

## To rerun this code - use a virtual environment to adjust package versions

This avoids completely ruining the other project on your machine that might depend on other versions of packages used in this project.

- Installation: https://virtualenv.pypa.io/en/latest/installation.html
- Create a virtual environment within you project folder: 'virtualenv --system-site-packages -p python3 ./venv'
- To activate: 'source venv/bin/activate' -> (venv) should pop up to the left on the command line.
- Use 'which python' to confirm that the shell uses the python inside venv/. Use 'pip freeze' to see the current package versions used in the virtual environment.
- To deactivate: 'deactivate'

## Dependencies

- Tensorflow 2.1.0
- Keras 2.1.6 (but I actually use tensorflow.keras, which comes as version 2.2.4-tf with tensorflow 2.1.0)

If updating: use e.g. 'pip install --upgrade tensorflow'