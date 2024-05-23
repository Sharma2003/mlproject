from setuptools import find_packages, setup
from typing import List

Hypen = '-e.'
def get_reqirements(file_path:str)->List[str]:
    "This function will return lists of requirements"

    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements = [req.replace("\n"," ") for req in requirements]

        if Hypen in requirements:
            requirements.remove(Hypen)


    return requirements
    
setup(
name = "mlproject",
version = '0.0.1',
author = "Tejash",
author_name = "tejassharma1820@gmail.com",
packages = find_packages(),
install_requires = get_reqirements('requirements.txt')
)