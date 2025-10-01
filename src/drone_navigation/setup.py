from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'drone_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        ('share/' + package_name + '/maps', glob('maps/*.pcd'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aryan',
    maintainer_email='aryan@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fov_node = drone_navigation.fov:main',
            'obstacles_node = drone_navigation.obstacles:main',
            'rf_node = drone_navigation.rf_signals7:main',
            'apf_node = drone_navigation.apf:main',
            'modified_apf_node = drone_navigation.modified_apf:main',
            'camera_node = drone_navigation.cam_control:main',
            'height_node = drone_navigation.height2:main',
        ],
    },
)
