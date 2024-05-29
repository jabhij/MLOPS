"""
    setup.py - responsible for machine learning application as a package that
    can be used in different other projects. Additioanlly, we can also deploy
    in 'Python Pi Pi'. From where anybody can use it and do the installation.
"""

from setuptools import find_packages, setup     # Accessing setup Packages
from typing import List

hyphen_e_dot = '-e .'


def get_requirements(file_path:str)->List[str]:
    '''
        This function returns the list of requirements!
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", " ") for req in requirements]

        if hyphen_e_dot in requirements:        # To remove '-e .' from reading
            requirements.remove(hyphen_e_dot)
            
    return requirements

# Basic Information of the Project
setup(
    name='MLOPS', 
    version='0.0.1',        # Current version that can be modified for changes in entire project
    author='Abhishek Jaiswal',
    author_email='jaisabhii@gmail.com',
    packages=find_packages(),       
    install_requires=get_requirements('requirements.txt')     # All basic requirements and will get automatically installed
)