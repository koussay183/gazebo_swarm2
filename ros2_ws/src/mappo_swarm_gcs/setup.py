from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mappo_swarm_gcs'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['README.md']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='koussay183',
    maintainer_email='web@slamatravel.com',
    description='Ground Control Station nodes for MAPPO drone swarm.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission_supervisor = mappo_swarm_gcs.mission_supervisor_node:main',
            'swarm_monitor = mappo_swarm_gcs.swarm_monitor_node:main',
            'emergency_control = mappo_swarm_gcs.emergency_control_node:main',
            'swarm_logger = mappo_swarm_gcs.logging_node:main',
        ],
    },
)
