from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mappo_swarm_training'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='koussay183',
    maintainer_email='web@slamatravel.com',
    description='MAPPO training package for the drone swarm.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_mappo = mappo_swarm_training.train_mappo:main',
            'evaluate_policy = mappo_swarm_training.evaluate_policy:main',
        ],
    },
)
