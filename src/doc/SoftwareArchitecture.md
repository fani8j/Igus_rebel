# Overview

The structure of the igus ReBeL control software follows the general ROS2 architecture. The following graph describes how components in the software interact with each other:

![Igus Rebel control software structure](./img/igusRebelSoftware.drawio.png)


The core elements are the planner, using Moveit2, and the ROS2 controller. There are two options: running an actual igus ReBeL robot or a robot in a Gazebo simulation environment. 

The package *igus_rebel* contains the configuration files for the ROS2 controller and the CRI hardware interface to control the real robot. The package *igus_rebel_moveit_config* contains configuration files for Moveit2, and a virtual controller to control a simulation. 

The robot model is stored in the package *igus_rebel_description*. This package also contains declarations of hardware plugins and gazebo plugins.