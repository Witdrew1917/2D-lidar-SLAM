cmake_minimum_required(VERSION 3.5.1)
project(map)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    sensor_msgs
    message_generation
)

catkin_python_setup()

add_message_files(
    FILES
    Landmark.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
)

catkin_package(
    CATKIN_DEPENDS message_runtime
)
catkin_install_python(PROGRAMS scripts/landmark_parser.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

