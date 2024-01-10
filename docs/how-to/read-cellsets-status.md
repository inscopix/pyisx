# Read status of a cell in a CellSet

!!! note "Assumptions"
    This section assumes you: 

    1. have the API installed, 
    2. have imported it using `import isx` 
    3. have a ISXD CellSet accessible on your local file system at `cellset.isxd`




Once a cell set is opened using:


```python
cell_set = isx.CellSet.read("cellset.isxd")
```
we can read the status of the first cell using:

```python
cell_status = cell_set.get_cell_status(0)
```

where `cell_status` is a string that is one of the following:


| Status| |
|---- | -- |
| `accepted` | This cell has been explicitly marked as accepted |
| `rejected` | This cell has been explicitly marked as rejeected |
| `undecided` | Default status of cells |

!!! warning "Indexing"
    Note that python indexes by 0, so the first frame is at index 0, and the the second frame is at index 1, and so on. 