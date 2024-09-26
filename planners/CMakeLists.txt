cmake_minimum_required(VERSION 3.5.1)
project(planners)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    message_generation
)

catkin_python_setup()

add_message_files(
    FILES
    AckermannDrive.msg
    AckermannDriveStamped.msg
)

generate_messages(
    DEPENDENCIES
    std_msgs
)

catkin_package(
    CATKIN_DEPENDS message_runtime
)
catkin_install_python(PROGRAMS scripts/square_planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

