from setuptools import setup
setup(name='siesta_python',
      version='1.0',
      description='Module for interfacing with the SIESTA DFT code in python',
      url='',
      author='Aleksander Bach Lorentzen',
      author_email='aleksander.bl.mail@gmail.com',
      license='MIT',
      packages=['siesta_python'],
      zip_safe=False,
      install_requires= ["numpy","numba", "seekpath", "spglib", "sisl","scipy"])
      
