# ieec

## Local setup

### Requirements
- Ubunutu 24.04
- ROS 2 Jazzy Jalisco

### Install

```bash
sudo apt update && sudo apt install ros-jazzy-rmw-zenoh-cpp -y
mkdir ~/ws_ieec/src -p
cd ~/ws_ieec/src
git clone https://github.com/intrinsic-dev/ieec
vcs import . < ieec/ieec.repos --recursive
rosdep install --from-paths src --ignore-src --rosdistro jazzy
source /opt/ros/jazzy/setup.bash
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install
```

### Launch

```bash
ros2 launch ieec_bringup ieec_gz_bringup.launch.py 
```