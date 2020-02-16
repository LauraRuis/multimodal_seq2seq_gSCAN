import json
import numpy as np
from typing import Dict, List, Union
import logging
import argparse

parser = argparse.ArgumentParser(description="Parse Grounded SCAN")

# General arguments
parser.add_argument("--dataset_path", type=str, default="./data/compositional_splits/dataset.txt", help="path to data",
                    required=True)
parser.add_argument("--output_file", type=str, default="parsed_dataset.txt")
parser.add_argument("--save_data", dest="save_data", default=False, action="store_true")

flags = vars(parser.parse_args())

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger(__name__)


def parse_sparse_situation(situation_representation: dict, grid_size: int) -> np.ndarray:
    """
    Each grid cell in a situation is fully specified by a vector:
    [_ _ _ _ _ _ _   _       _      _       _   _ _ _ _]
     1 2 3 4 r g b circle square cylinder agent E S W N
     _______ _____ ______________________ _____ _______
       size  color        shape           agent agent dir.
    :param situation_representation: data from dataset.txt at key "situation".
    :param grid_size: int determining row/column number.
    :return: grid to be parsed by computational models.
    """
    num_object_attributes = len([int(bit) for bit in situation_representation["target_object"]["vector"]])
    # Object representation + agent bit + agent direction bits (see docstring).
    num_grid_channels = num_object_attributes + 1 + 4

    # Initialize the grid.
    grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)

    # Place the agent.
    agent_row = int(situation_representation["agent_position"]["row"])
    agent_column = int(situation_representation["agent_position"]["column"])
    agent_direction = int(situation_representation["agent_direction"])
    agent_representation = np.zeros([num_grid_channels], dtype=np.int)
    agent_representation[-5] = 1
    agent_representation[-4 + agent_direction] = 1
    grid[agent_row, agent_column, :] = agent_representation

    # Loop over the objects in the world and place them.
    for placed_object in situation_representation["placed_objects"].values():
        object_vector = np.array([int(bit) for bit in placed_object["vector"]], dtype=np.int)
        object_row = int(placed_object["position"]["row"])
        object_column = int(placed_object["position"]["column"])
        grid[object_row, object_column, :] = np.concatenate([object_vector, np.zeros([5], dtype=np.int)])
    return grid


def data_loader(file_path: str) -> Dict[str, Union[List[str], np.ndarray]]:
    """
    Loads grounded SCAN dataset from text file and ..
    :param file_path: Full path to file containing dataset (dataset.txt)
    :returns: dict with as keys all splits and values list of example dicts with input, target and situation.
    """
    with open(file_path, 'r') as infile:
        all_data = json.load(infile)
        grid_size = int(all_data["grid_size"])
        splits = list(all_data["examples"].keys())
        logger.info("Found data splits: {}".format(splits))
        loaded_data = {}
        for split in splits:
            loaded_data[split] = []
            logger.info("Now loading data for split: {}".format(split))
            for data_example in all_data["examples"][split]:
                input_command = data_example["command"].split(',')
                target_command = data_example["target_commands"].split(',')
                situation = parse_sparse_situation(situation_representation=data_example["situation"],
                                                   grid_size=grid_size)
                loaded_data[split].append({"input": input_command,
                                           "target": target_command,
                                           "situation": situation.tolist()})  # .tolist() necessary to be serializable
            logger.info("Loaded {} examples in split {}.\n".format(len(loaded_data[split]), split))
    return loaded_data


if __name__ == "__main__":
    data = data_loader(flags["dataset_path"])
    if flags["save_data"]:
        with open(flags["output_file"], 'w') as outfile:
            json.dump(data, outfile, indent=4)
