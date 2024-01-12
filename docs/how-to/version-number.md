To determine the version number of installed code, use:

```python
import isx
isx.__version__
```

Versioning follows [semantic versioning](https://semver.org/). 

!!! warning "Version numbers for developers"
    If you installed the code from git sources, for the purposes of developing this code further, the version number will always be `0.0.0.dev`. 

    Version numbers are derived from git tags of the repository. Pushing a new tag triggers an action that publishes a new version of the code on PyPI. 