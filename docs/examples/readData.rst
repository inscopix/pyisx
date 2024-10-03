.. _exampleReadData:

Reading Data
============

This section demonstrates how the :code:`isx` package can be used to read data from Inscopix files.
Refer to :ref:`this table <fileTypes>` for reference on the Inscopix file types and read support.

.. note::
   
   The following sections assume the :code:`isx` package has been imported, i.e., :code:`import isx`

Microscope & Behavior Movies
----------------------------

`Microscope Movie` and `Behavior Movie` file types can be read using the :code:`isx.Movie` class. 

.. code-block:: python

    # Open the movie for reading
    movie = isx.Movie.read("movie.isxd")

Movie objects have timing and spacing properties which can be accessed:


.. code-block:: python

    # Timing information can be accessed using:
    movie.timing

    # The number of frames can be accessed using:
    movie.timing.num_samples

    # The time period can be accessed using:
    movie.timing.period.secs_float

    # Spacing information can be accessed using:
    movie.spacing

    # The dimensions of the frame can be accessed using:
    # which will return a 2-tuple containing the dimensions of the frame
    # (num_rows, num_cols), or (height, width)
    movie.spacing.num_pixels

Frames from movies can be read into memory:

.. code-block:: python

    # The pixel data type can accessed using:
    movie.data_type

    # Read every frame of the movie, and process it
    for i in range(movie.timing.num_samples):
        frame = movie.get_frame_data(i)
        
        # Process frame
        ...

.. warning::
    
    It's recommended to read one frame into memory at a time to prevent out of memory errors.

.. note::
    
    Python indexes by 0, so the first frame is at index 0, and the the second frame is at index 1, and so on.

Cell Sets
----------

`Cell Set` file types can be read using the :code:`isx.CellSet` class. 

.. code-block:: python

    # Open the cell set for reading
    cell_set = isx.CellSet.read("cell_set.isxd")

Similar to movies, cell sets have timing and spacing properties which can be accessed.
These properties are derived from the parent movie which generated the cell set.

.. code-block:: python

    # Timing information can be accessed using:
    cell_set.timing

    # The number of frames can be accessed using:
    cell_set.timing.num_samples

    # The time period can be accessed using:
    cell_set.timing.period.secs_float

    # Spacing information can be accessed using:
    cell_set.spacing

    # The dimensions of the frame can be accessed using:
    # which will return a 2-tuple containing the dimensions of the frame
    # (num_rows, num_cols), or (height, width)
    cell_set.spacing.num_pixels

Cell data from cell sets can be read into memory:

.. code-block:: python

    # The number of cells in the cell set can accessed using:
    cell_set.num_cells

    # Read data of every cell in the cell set, and process it
    for i in range(cell_set.num_cells):
        # Get the cell name
        name = cell_set.get_cell_name(i)

        # Get the temporal activity trace of a cell
        trace = cell_set.get_cell_trace(i)

        # Get the spatial footprint of a cell
        footprint = cell_set.get_cell_image_data(i)
        
        # Process cell data
        ...

Event Sets
----------

`Event Set` file types can be read using the :code:`isx.EventSet` class. 

.. code-block:: python

    # open the event set for reading
    event_set = isx.EventSet.read("event_set.isxd")

Similar to cell sets, event sets have timing properties which can be accessed.
These properties are derived from the parent cell set which generated the event set.

.. code-block:: python

    # Timing information can be accessed using:
    event_set.timing

    # The number of frames can be accessed using:
    event_set.timing.num_samples

    # The time period can be accessed using:
    event_set.timing.period.secs_float

Cell data from event sets can be read into memory:

.. code-block:: python

    # Read data of every cell in the event set, and process it
    for i in range(event_set.timing.num_samples):
        # Get the cell name
        name = event_set.get_cell_name(i)

        # Get the event timestamps and amplitudes of a cell
        offsets, amplitudes = event_set.get_cell_data(i)

        # Process cell data
        ...

Vessel Sets
-----------

`Vessel Set` file types can be read using the :code:`isx.VesselSet` class. 

.. code-block:: python

    # Open the vessel set for reading
    vessel_set = isx.VesselSet.read("vessel_set.isxd")

Similar to movies, vessel sets have timing and spacing properties which can be accessed.
These properties are derived from the parent movie which generated the vessel set.

.. code-block:: python

    # Timing information can be accessed using:
    vessel_set.timing

    # The number of frames can be accessed using:
    vessel_set.timing.num_samples

    # The time period can be accessed using:
    vessel_set.timing.period.secs_float

    # Spacing information can be accessed using:
    vessel_set.spacing

    # The dimensions of the frame can be accessed using:
    # which will return a 2-tuple containing the dimensions of the frame
    # (num_rows, num_cols), or (height, width)
    vessel_set.spacing.num_pixels

Vessel data from vessel sets can be read into memory:

.. code-block:: python

    # The number of vessels in the vessel set can be accessed using:
    vessel_set.num_vessels

    # The standard deviation projection image of the parent movie
    # This is the same for every vessel
    image = vessel_set.get_vessel_image_data(0)

    # Read data of every vessel in the vessel set, and process it
    for i in range(vessel_set.num_vessels):
        # Get the vessel name
        name = vessel_set.get_vessel_name(i)

        # Get the vessel status
        status = vessel_set.get_vessel_status(i)

        # For vessel diameter vessel sets, get the following data:
        # The vessel diameter trace
        trace = vessel_set.get_vessel_trace_data(i)
        # The direction of rbc velocity trace
        center_trace = vessel_set.get_vessel_center_trace_data(i)
        # The vessel line points
        line = vessel_set.get_vessel_line_data(i)

        # For rbc velocity vessel sets, get the following data:
        # The rbc velocity trace
        trace = vessel_set.get_vessel_trace_data(i)
        # The direction of rbc velocity trace
        direction_trace = vessel_set.get_vessel_direction_trace_data(i)
        # The first correlation heatmap of the vessel
        corr = vessel_set.get_vessel_correlations_data(i, 0)
        # The vessel box points
        box = vessel_set.get_vessel_line_data(i)

        # Process vessel data
        ...

GPIO & IMU
----------
 
`GPIO` and `IMU` file types can be read using the :code:`isx.GpioSet` class. 

.. code-block:: python

    # open the gpio set for reading
    gpio_set = isx.GpioSet.read("signals.gpio")

Gpio sets have timing properties which can be accessed.

.. code-block:: python

    # Timing information can be accessed using:
    gpio_set.timing

    # The number of frames can be accessed using:
    gpio_set.timing.num_samples

    # The time period can be accessed using:
    gpio_set.timing.period.secs_float

Signal data from gpio sets can be read into memory:

.. code-block:: python

    # Number of channels (i.e., signals) can be accessed using:
    gpio_set.num_channels

    # Read data of every cell in the cell set, and process it
    for i in range(gpio_set.num_channels):
        # Get the channel name
        name = gpio_set.get_channel_name(i)

        # Get the signal timestamps and amplitudes of a channel
        offsets, amplitudes = gpio_set.get_channel_data(i)

        # Process signal data
        ...
