## Results

The result plots and csv-files of ROS messages are contained in this file.

In addition, some helpful scripts are:

  - bag2csv.sh: a shell script which transforms rosbag files into .csv files
  example: './bag2csv.sh bagfile.bag'

  - add_timestamps.py: a short python3 program which adds timestamps to the outermost column of the .csv-files, since rosbags stores time as nanoseconds from 1.jan 1970. It basically just takes in the data, and adds a timestamp counting number of seconds since the initial message ROS time. The paths of the files are declared inside the script.
  example: 'python3 add_timestamps.py' 

  - add_prefix.py: a short python3 program which renames .csv files with whatever prefix is given as commandline argument. NB: time_adder will not work afterwards
  example: 'python3 add_prefix.py --p RL' 
