# TL;DR
This file reads a fully specified gSCAN dataset into a representation that can be processed by numerical models.
 
Unzip one of the data folders in `../data`, e.g. `compositional_splits`, or use the demo dataset (`../data/demo_dataset/dataset.txt`).
 
To obtain the parsed dataset in a file named "parsed_dataset.txt", run:
```bash
>>> python -m pip install numpy
>>> python read_gscan.py --dataset_path=../data/compositional_splits/dataset.txt --save_data --output_file=parsed_dataset.txt
```

Another possibility is to import the function `data_loader` in another project and use the data directly in a model.

# Prelims
```python -m pip install numpy```

NB: if you want to load `./data/compositional_splits/dataset.txt` note that it is 2 GB.

# Load grounded SCAN data set.
Specify the path to the `dataset.txt` file as argument `--dataset_path`.

If you want to save the dataset in a new file with parsed situation representation, set the flag `--save_data`.

# Background information

In `./data/compositional_splits/dataset.txt` the situation of each data example is represented sparsely,
meaning a list of placed objects is saved as opposed to the full grid size by grid size grid.
This code parses that representation back into a grid that can be used by computational models.

Each grid cell in a situation is fully specified by a vector of size 15.
The first 10 dimensions specify the object (if there is one). The first 4 of those is a one-hot vector
specifying the size of the object (so [0 0 1 0] for size 3), the second 3 dimensions specify
the object color (e.g. [0 1 0] for green), then the shape ([0 1 0] for a square).

Finally the last five dimensions specify the agent (at the cell it is currently):
The first dimension is 1 if there's an agent, 0 if not, the last four dimensions specify the
direction the agent is looking in ([1 0 0 0] fo east).

**Dimensions**:

[size 1, size 2, size 3, size 4, green, red, blue, circle, square, cylinder, agent, east, south, west, north]
