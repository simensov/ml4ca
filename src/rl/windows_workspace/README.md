
## How to use

## Tips and tricks
    - A known error is, when trying to launch a simulator and load its configuration (if the simulator hasn't already been opened) is: 
        - py4j.protocol.Py4JNetworkError: An error occurred while trying to connect to the Java server (127.0.0.1:25344)

        This is no big problem; the simulator windows opens, but the configuration is not loaded and the Python script terminates. Just load the configuration manually in the simulator window, and launch the Python script when completed. 