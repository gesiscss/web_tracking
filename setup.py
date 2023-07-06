from setuptools import setup, find_packages

setup(name='web_tracking',
      version='0.1',
      description='A Web browsing trajectory analysis library.',
      url='https://github.com/gesiscss/web_tracking',
      author='Orkut Karacalik',
      author_email='okaracalik12@gmail.com',
      python_requires='>=3.6',
      license='gpl',
      packages=find_packages(include=["web_tracking", "web_tracking.*"]),
      install_requires=[
          'pandas==1.0.5',
          'matplotlib==3.2.2',
          'statsmodels==0.11.1',
          'scipy==1.10.0'
      ],
      zip_safe=False)
