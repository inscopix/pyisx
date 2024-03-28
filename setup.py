# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['isx']

package_data = \
{'': ['*']}

install_requires = \
['beartype>=0.15.0', 'importlib-metadata>=7.0.1,<8.0.0', 'numpy>=1.26.2']

extras_require = \
{'dev': ['ipykernel>=6.20.1',
         'debugpy==1.6',
         'matplotlib>=3.8.2',
         'poetry2setup>=1.1.0,<2.0.0'],
 'docs': ['mkdocs>=1.4.2,<2.0.0',
          'mkdocs-material-extensions>=1.1.1,<2.0.0',
          'mkdocs-material>=9.0.9,<10.0.0',
          'mkdocstrings>=0.24.0,<0.25.0',
          'mkdocstrings-python>=1.7.5,<2.0.0',
          'mkdocs-git-revision-date-localized-plugin>=1.2.2,<2.0.0',
          'mkdocs-git-committers-plugin-2>=2.2.3,<3.0.0'],
 'test': ['pytest>=7.2.0',
          'poetry2setup>=1.1.0,<2.0.0',
          'requests>=2.31.0,<3.0.0']}

setup_kwargs = {
    'name': 'isx',
    'version': '0.0.0.dev0',
    'description': 'Python-based ISXD file reader',
    'long_description': '# isx: pure-python API to read Inscopix data\n\n![](https://github.com/inscopix/py_isx/actions/workflows/main.yml/badge.svg) \n![](https://img.shields.io/pypi/v/isx)\n\nThis is a pure-python API to read Inscopix ISXD files. \n\n\n## Documentation\n\n[Read the documentation](https://inscopix.github.io/py_isx/)\n\n## Support\n\n|  File type | Support |\n|  --------- | ------- |\n| ISXD CellSet   | ✅ |\n| ISXD Movie   | ✅ |\n| ISXD Movie (multi-plane)   | ❌ |\n| ISXD Movie (dual-color)   | ❌ |\n| GPIO data   | ❌ |\n| ISXD Events   | ❌ |\n| ISXD VesselSet   | ❌ |\n\n\n## Install\n\n### Poetry\n\n```bash\npoetry add isx\n```\n\n### pip\n\n\n```bash\npip install isx\n```\n\n## Caution\n\nThis is a work in progress, and all reading functions in the IDPS Python API are not supported yet. \n\n\n## Testing\n\nThis code is tested using GitHub Actions on the following python\nversions:\n\n- 3.9\n- 3.10\n- 3.11\n- 3.12\n',
    'author': 'Inscopix, Inc.',
    'author_email': 'support@inscopix.com',
    'url': 'https://github.com/inscopix/py_isx',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'python_requires': '>=3.9,<4.0',
}


setup(**setup_kwargs)

