# ROS 2 distribution: humble (primary, Ubuntu 22.04) or jazzy (Ubuntu 24.04)
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}
ARG ROS_DISTRO

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_WS=/ros2_ws

RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    curl \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

COPY ./src/ ${ROS_WS}/src/
WORKDIR ${ROS_WS}

RUN apt-get update \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src -r -y --rosdistro ${ROS_DISTRO} \
    && rm -rf /var/lib/apt/lists/*

RUN bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install"

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc \
    && echo "source ${ROS_WS}/install/setup.bash" >> ~/.bashrc

CMD ["bash"]
