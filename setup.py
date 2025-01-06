from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path:str) -> List[str]:
    
    """ 
    This function will return a list of requirements 
    """

    HYPHEN_E_DOT = '-e .'
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [requirement.replace("\n","")for requirement in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements
        


setup(
    name="mlproject",
    version='1.0.0',
    author='Faheem',
    author_email='adahm7114@gmail.com',
    packages=find_packages(where='src'),
    package_dir = {"":'src'},
    install_requires=get_requirements('requirements.txt'),
)