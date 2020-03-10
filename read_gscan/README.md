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

[size 1, size 2, size 3, size 4, circle, square, cylinder, green, red, yellow, blue, agent, east, south, west, north]

# Demonstration

To get a feel of the data used in the paper, we can look at the data in the folder `../data/demo_dataset/..`. This dataset is highly simplified in terms of grid size, vocabulary, and number of examples, but the ideas are the same. When opening `../data/demo_dataset/dataset.txt` we can see that the first example if we follow the keys "examples" and "situational_1" is the following: 

<details open>
<summary>The first data example in the split called "situational_1" (i.e., novel direction) set. Click to open/close.</summary>
<p>
 
```javascript
{
                "command": "walk,to,a,red,circle",
                "meaning": "walk,to,a,red,circle",
                "derivation": "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP;T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:red,NT:JJ -> red,T:circle,NT:NN -> circle",
                "situation": {
                    "grid_size": 4,
                    "agent_position": {
                        "row": "2",
                        "column": "3"
                    },
                    "agent_direction": 0,
                    "target_object": {
                        "vector": "1000101000",
                        "position": {
                            "row": "3",
                            "column": "2"
                        },
                        "object": {
                            "shape": "circle",
                            "color": "red",
                            "size": "1"
                        }
                    },
                    "distance_to_target": "2",
                    "direction_to_target": "sw",
                    "placed_objects": {
                        "0": {
                            "vector": "1000101000",
                            "position": {
                                "row": "3",
                                "column": "2"
                            },
                            "object": {
                                "shape": "circle",
                                "color": "red",
                                "size": "1"
                            }
                        },
                        "1": {
                            "vector": "0010011000",
                            "position": {
                                "row": "2",
                                "column": "1"
                            },
                            "object": {
                                "shape": "square",
                                "color": "red",
                                "size": "3"
                            }
                        },
                        "2": {
                            "vector": "0100010001",
                            "position": {
                                "row": "1",
                                "column": "2"
                            },
                            "object": {
                                "shape": "square",
                                "color": "blue",
                                "size": "2"
                            }
                        },
                        "3": {
                            "vector": "0001010100",
                            "position": {
                                "row": "1",
                                "column": "1"
                            },
                            "object": {
                                "shape": "square",
                                "color": "green",
                                "size": "4"
                            }
                        }
                    },
                    "carrying_object": null
                },
                "target_commands": "turn left,turn left,walk,turn left,walk",
                "verb_in_command": "walk",
                "manner": "",
                "referred_target": " red circle"
            }
```

</p>
</details>

This data example contains the *"command"*, or input  instruction, 'walk to the red circle', that in this case based on the situation maps to the target command sequence of *"target_commands"*: "turn left,turn left,walk,turn left,walk". The data example contains the situation representation, or world state, at the key *"situation"*, and it also contains some additional information that is useful in parsing it back to the representation it was generated from, namely the *"derivation"* containing the depth-first extracted constituency tree, the *"meaning"* containing the semantic meaning of the input instruction. This is only useful if we would have generated the benchmark with nonsensical words, in that case we would need a semantic representation that can be parsed by humans. 

This example is visualized by the following animation:

![demo_example](https://raw.githubusercontent.com/LauraRuis/multimodal_seq2seq_gSCAN/master/data/demo_dataset/walk_to_a_red_circle/situation_1/movie.gif)

Is we want to parse the demo_dataset independent from the code GroundedScan, this particular example in represented by:

<details open>
<summary>The parsed data example. Click to open/close, explanation below. </summary>
<p>
 
```javascript
{
            "input": [
                "walk",
                "to",
                "a",
                "red",
                "circle"
            ],
            "target": [
                "turn left",
                "turn left",
                "walk",
                "turn left",
                "walk"
            ],
            "situation": [
                [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ]
                ],
                [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ]
                ],
                [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        1,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        0,
                        0
                    ]
                ],
                [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ]
                ]
            ]
        }
```

</p>
</details>

The parsed data only retains the important keys for training a computational model, namely the input pair *"input"* and *"situation"*, and the target sequence *"target"*. The situation representation is instead of a sparse representation of the available objects represented by the, in this case, 4 x 4 x 15 sized matrix of the grid world, where the 15 dimensions are the following:

[size 1, size 2, size 3, size 4, circle, square, red, green, yellow, blue, agent, east, south, west, north]

In the first vector you can see the representation of the green square of size 4 in row 1 and column 1 (starting from 0) of the world. Namely the following vector: `[0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]`.

