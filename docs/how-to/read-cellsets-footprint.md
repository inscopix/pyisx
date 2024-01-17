# Read footprint of a cell in a CellSet

!!! note "Assumptions"
    This section assumes you: 

    1. have the API installed, 
    2. have imported it using `import isx` 
    3. have a ISXD CellSet accessible on your local file system at `cellset.isxd`


Once a cell set is opened using:


```python
cell_set = isx.CellSet.read("cellset.isxd")
```
we can read the footprint of the first cell using:

```python
footprint = cell_set.get_cell_image_data(0)
```

`footprint` now contains a 2D [numpy array](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) with the image data for the first frame. 



!!! warning "Indexing"
    Note that python indexes by 0, so the first frame is at index 0, and the the second frame is at index 1, and so on. 