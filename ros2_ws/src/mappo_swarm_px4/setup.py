from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mappo_swarm_px4'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['README.md']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='koussay183',
    maintainer_email='web@slamatravel.com',
    description='PX4 offboard control integration for MAPPO multi-agent drone swarm.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mappo_px4_rl_node = mappo_swarm_px4.mappo_px4_rl_node:main',
            'px4_state_listener = mappo_swarm_px4.px4_state_listener:main',
            'swarm_diagnostics = mappo_swarm_px4.diagnostics:main',
        ],
    },
)
