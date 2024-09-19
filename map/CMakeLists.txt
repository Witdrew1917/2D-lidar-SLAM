cmake_minimum_required(VERSION 3.5.1)
project(map)
find_package(catkin REQUIRED rospy sensor_msgs)
catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS scripts/landmark_parser.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
