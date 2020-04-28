## Results

The result plots and csv-files of ROS messages are contained in this file.

In addition, two helpful scripts are:

  - bag2csv.sh: a shell script which transforms rosbag files into .csv files
  example: './bag2csv.sh bagfile.bag'

  - time_adder.py: a short python3 program which adds timestamps to the outermost column of the .csv-files, since rosbags stores time as nanoseconds from 1.jan 1970. It basically just takes in the data, and adds a timestamp counting number of seconds since the initial message ROS time. The paths of the files are declared inside the script.
  example: 'time_adder.py' 
