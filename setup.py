# -*- coding: utf-8 -*-
# @Author: Shuang0420
# @Date:   2017-12-12 11:47:28
# @Last Modified by:   Shuang0420
# @Last Modified time: 2017-12-12 11:47:28

'''
Build the python-aiml Py2/Py3 Chinese version package
'''

from setuptools import setup

from rankbot import VERSION


setup_args = dict( name="rankbot",
                   version=VERSION,
                   author="Shuang0420",
                   author_email="sxu1@alumni.cmu.edu",

                   description="Dual LSTM Encoder",
                   long_description="""implements the Dual LSTM Encoder model from The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems.
""",
                   url="https://github.com/Shuang0420/Dual-LSTM-Encoder-Rank-Model.git",
                   platforms=["any"],
                   classifiers=["Development Status :: 1 - Beta",
                                "Environment :: Console",
                                "Intended Audience :: Developers",
                                "Programming Language :: Python",
                                "Programming Language :: Python :: 3",
                                "Programming Language :: Python :: 3.4",
                                "Programming Language :: Python :: 3.5",
                                "Programming Language :: Python :: 3.6",
                                "License :: OSI Approved :: BSD License",
                                "Operating System :: OS Independent",
                                "Topic :: Communications :: Chat",
                                "Topic :: Scientific/Engineering :: Artificial Intelligence"
                                ],

                   install_requires = [ 'setuptools',
                                        ],

                   packages=[ "rankbot"],
                   #    package_dir = { 'aiml': 'aiml',
                   #                    'aiml.script' : 'aiml/script' },

                   include_package_data = False,       # otherwise package_data is not used
                   # package_data={ 'aiml': ['botdata/*',
                   #                         ]},

                   # entry_points = { 'console_scripts': [
                   #     'aiml-validate = aiml.script.aimlvalidate:main',
                   #     'aiml-bot = aiml.script.bot:main',
                   # ]},

                   # test_suite = 'test.__main__.load_tests',

                   #    data_files=[
                   #        (package_prefix, glob.glob("aiml/self-test.aiml")),
                   #        (package_prefix, glob.glob("*.txt")),
                   #    ],
                   )

if __name__ == '__main__':
    setup( **setup_args )
