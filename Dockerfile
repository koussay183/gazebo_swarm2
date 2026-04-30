FROM osrf/ros:jazzy-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ── Ajouter le repo OSRF Gazebo ────────────────────────────────────────────
RUN apt-get update && apt-get install -y curl lsb-release gnupg \
    && curl https://packages.osrfoundation.org/gazebo.gpg \
       --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
       http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
       > /etc/apt/sources.list.d/gazebo-stable.list \
    && rm -rf /var/lib/apt/lists/*

# ── Gazebo Harmonic + bridge ROS2 ──────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    gz-harmonic \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-interfaces \
    ros-jazzy-visualization-msgs \
    python3-pip \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances PX4 SITL ───────────────────────────────────────────────────
# Outils de base + dépendances PX4
RUN apt-get update && apt-get install -y \
    git cmake ninja-build ccache \
    python3-jinja2 python3-jsonschema python3-toml \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl \
    libfuse2 wget unzip \
    && rm -rf /var/lib/apt/lists/*

# ── Cloner et compiler PX4 SITL ───────────────────────────────────────────
# ATTENTION : cette étape prend ~20-30 minutes lors du premier build Docker.
# Le binaire px4_sitl_default est ensuite mis en cache.
ENV PX4_PATH=/opt/px4
RUN mkdir -p $PX4_PATH && \
    git clone --recursive --depth=1 \
        https://github.com/PX4/PX4-Autopilot.git \
        $PX4_PATH/PX4-Autopilot

# Installation des dépendances PX4 (script officiel, sans NuttX)
RUN cd $PX4_PATH/PX4-Autopilot && \
    bash ./Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools

# Compilation PX4 SITL (cible Gazebo Harmonic)
RUN cd $PX4_PATH/PX4-Autopilot && \
    make px4_sitl -j$(nproc)

ENV PX4_BIN=$PX4_PATH/PX4-Autopilot/build/px4_sitl_default/bin/px4

# ── Python packages ────────────────────────────────────────────────────────
RUN pip3 install --break-system-packages \
    torch \
    numpy \
    gymnasium \
    mavsdk

# ── Symlink python3.12 (appelé explicitement dans run.sh) ─────────────────
RUN ln -sf /usr/bin/python3 /usr/local/bin/python3.12

# ── Setup ROS2 au démarrage ────────────────────────────────────────────────
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc

WORKDIR /workspace
