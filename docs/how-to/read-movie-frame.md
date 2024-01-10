# Read a frame from a movie

!!! note "Assumptions"
    This section assumes you: 

    1. have the API installed, 
    2. have imported it using `import isx` 
    3. have a ISXD movie accessible on your local file system at `movie.isxd`

First, open the movie for reading using:


```python
movie = isx.Movie.read("movie.isxd")

```

To read the first frame:

```python
frame = movie.get_frame_data(0) 
```

`frame` contains a [numpy array](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) with the frame data for the first frame. 

!!! warning "Indexing"
    Note that python indexes by 0, so the first frame is at index 0, and the the second frame is at index 1, and so on. 