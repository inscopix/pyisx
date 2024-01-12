# Read movie timing information

!!! note "Assumptions"
    This section assumes you: 

    1. have the API installed, 
    2. have imported it using `import isx` 
    3. have a ISXD movie accessible on your local file system at `movie.isxd`

Once a movie is opening using:


```python
movie = isx.Movie.read("movie.isxd")
```

Spacing information can be accessed using:

```python
movie.spacing
```

The dimensions of the frame can be accessed using:

```python
movie.spacing.num_pixels
```

which will return a 2-tuple containing the dimensions of the frame.