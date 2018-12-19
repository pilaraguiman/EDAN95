from setuptools import setup, find_packages

setup(
    name='rlagent',
    version=0.0.1,
    description='',
    keywords='',
    author='Erik Gärtner',
    author_email='erik.gartner@math.lth.se',
    url='',
    license='',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    include_package_data=False,
    entry_points={
        'console_scripts': [
            'rlagent=rlagent.main'
        ],
    },
)
