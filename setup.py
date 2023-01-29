from setuptools import find_packages, setup
import ya_cup_2022


module_name = 'ya_cup_2022'


def parse_requirements(filename):
    requirements = list()
    with open(filename) as f:
        for line in f:
            requirements.append(line.rstrip())
    return requirements


setup(
    name=module_name,
    version=ya_cup_2022.__version__,
    author=ya_cup_2022.__author__,
    author_email=ya_cup_2022.__email__,
    description=ya_cup_2022.__doc__,
    platforms='all',
    python_requires='~=3.7',
    packages=find_packages(exclude=['tests']),
    install_requires=parse_requirements('requirements.txt'),
    include_package_data=True,
)
