
# Writing Data

This section demonstrates how the `isx` package can be used to write data to Inscopix files.
Refer to :ref:`this table <fileTypes>` for reference on the Inscopix file types and write support.

::: {note}
The following sections assume the `isx` package has been imported, i.e., `import isx`
:::

## Microscope Movies

The `Microscope Movie` file type can be written using the `isx.Movie` class. 

```python
    # Open the movie for writing
    movie = isx.Movie.write('movie.isxd', timing, spacing, numpy.float32)

    for i in range(timing.num_samples):
        # Generate frame to write
        frame = ...

        # Write to file
        movie.set_frame_data(i, frame)
    
    # Flush to disk
    movie.flush()
```

::: {note}
Python indexes by 0, so the first frame is at index 0, and the the second frame is at index 1, and so on.
:::

## Cell Sets

The `Cell Set` file type can be written using the `isx.CellSet` class. 

```python
    # Open the movie for writing
    cell_set = isx.CellSet.write('cell_set.isxd', timing, spacing, numpy.float32)

    for i in range(timing.num_samples):
        # Generate cell footprint and trace to write
        image = ...
        trace = ...

        # Write to file
        cell_set.set_cell_data(i, image, trace, 'C{}'.format(i))
    
    # Flush to disk
    cell_set.flush()
```

## Event Sets

The `Event Set` file type can be written using the `isx.EventSet` class. 

```python
    # Open the event set for writing
    event_set = isx.EventSet.write('event_set.isxd', timing, cell_names)

    for i in range(cell_set.num_cells):
        # Generate event offsets and amplitudes
        offsets = ...
        amplitudes = ...

        # Write to file
        event_set.set_cell_data(c, offsets, amplitudes)
    
    # Flush to disk
    event_set.flush()
```

## Vessel Sets

The `Vessel Set` file type can be written using the `isx.VesselSet` class. 

For vessel diameter vessel sets:

```python
    # Open the vessel set for writing
    vessel_set = isx.VesselSet.write('vessel_set.isxd', timing, spacing, 'vessel diameter')

    # Generate a projection image of the parent movie
    image = ...
    for i in range(timing.num_samples):
        # Generate vessel diameter line, trace, and center trace to write
        line = ..
        trace = ...
        center_trace = ...

        # Write to file
        vessel_set.set_vessel_diameter_data(i, image, line, trace, center_trace, 'V{}'.format(i))
    
    # Flush to disk
    vessel_set.flush()
```

For RBC velocity vessel sets:

```python
    # Open the vessel set for writing
    vessel_set = isx.VesselSet.write('vessel_set.isxd', timing, spacing, 'rbc velocity')

    # Generate a projection image of the parent movie
    image = ...
    for i in range(timing.num_samples):
        # Generate rbc velocity line, trace, and direction trace, and correlation heatmaps to write
        line = ...
        trace = ...
        direction_trace = ...
        correlations_trace = ...

        # Write to file
        vessel_set.set_rbc_velocity_data(i, image, line, trace, direction_trace, correlations_trace, 'V{}'.format(i))
    
    # Flush to disk
    vessel_set.flush()
```
