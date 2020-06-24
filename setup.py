from setuptools import setup

setup(name='web_tracking',
      version='0.1',
      description='A Web browsing trajectory analysis library.',
      url='https://github.com/gesiscss/web_tracking',
      author='Orkut Karacalik',
      author_email='okaracalik12@gmail.com',
      license='gpl',
      packages=['web_tracking'],
      install_requires=[
          'pandas',
          'matplotlib',
          'statsmodels',
          'scipy'
      ],
      zip_safe=False)
