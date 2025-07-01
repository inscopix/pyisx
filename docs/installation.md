
# Installation

Pre-built binaries of this API can be installed from [PyPi](https://pypi.org/project/isx) for the supported platforms.

```bash
pip install isx
```

::: {attention}
Currently, pyisx is only supported for x86 architectures, which can be problematic, specifically on the newer Mac computers with Apple Silicon. For usage with Apple Silicon, the Rosetta software must be installed, and the Terminal app must be configured to use this software for automatic translation of x86 binaries to arm64. Read more [here](https://support.apple.com/en-us/102527) on how to configure Rosetta on Mac computers.
:::

## Supported Platforms

This package has been built and tested on the following operating systems, for python versions `3.9 - 3.12`:

|  OS | Version | Architecture |
|  --------- | ------- | ----- |
| macOS   | 13 | x86_64 |
| Ubuntu (Linux) | 20.04 | x86_64 |
| Windows | 11 | amd64 |

## Development (advanced)

In order to build the package locally, follow this [Development Guide](https://github.com/inscopix/pyisx?tab=readme-ov-file#development-guide).
