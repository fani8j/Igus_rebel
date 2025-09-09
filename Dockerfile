# Use the official ROS 2 Jazzy base image
FROM ros:jazzy

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_WS=/ros2_ws

# Install general dependencies
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    curl \
    ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
COPY ./src/ ${ROS_WS}/src/
WORKDIR ${ROS_WS}

RUN apt-get update 
RUN rosdep update 
RUN rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
RUN bash -c "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install"

# Source ROS 2 setup on container start
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
RUN echo "source ${ROS_WS}/install/setup.bash" >> ~/.bashrc

# Default to bash shell
CMD ["bash"]
