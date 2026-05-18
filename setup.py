from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'ur3e_realsense_handeye'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'docs'),
         glob('docs/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nishanth Sundaran',
    maintainer_email='sundharnishanth@gmail.com',
    description='Eye-to-hand calibration suite for UR3e + Intel RealSense D435i.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'calibration_node = ur3e_realsense_handeye.calibration_node:main',
            'board_generator = ur3e_realsense_handeye.board_generator:main',
        ],
    },
)
