import os
from typing import List
from typing import Tuple
import logging
from collections import defaultdict
from collections import Counter
import json
import torch
import numpy as np
import random
from gym import spaces
import gym

from GroundedScan.dataset import GroundedScan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("GroundedSCAN_learning")


class Vocabulary(object):
    """
    Object that maps words in string form to indices to be processed by numerical models.
    """

    def __init__(self, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>"):
        """
        NB: <PAD> token is by construction idx 0.
        """
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self._idx_to_word = [pad_token, sos_token, eos_token]
        self._word_to_idx = defaultdict(lambda: self._idx_to_word.index(self.pad_token))
        self._word_to_idx[sos_token] = 1
        self._word_to_idx[eos_token] = 2
        self._word_frequencies = Counter()

    def word_to_idx(self, word: str) -> int:
        return self._word_to_idx[word]

    def idx_to_word(self, idx: int) -> str:
        return self._idx_to_word[idx]

    def add_sentence(self, sentence: List[str]):
        for word in sentence:
            if word not in self._word_to_idx:
                self._word_to_idx[word] = self.size
                self._idx_to_word.append(word)
            self._word_frequencies[word] += 1

    def most_common(self, n=10):
        return self._word_frequencies.most_common(n=n)

    @property
    def pad_idx(self):
        return self.word_to_idx(self.pad_token)

    @property
    def sos_idx(self):
        return self.word_to_idx(self.sos_token)

    @property
    def eos_idx(self):
        return self.word_to_idx(self.eos_token)

    @property
    def size(self):
        return len(self._idx_to_word)

    @classmethod
    def load(cls, path: str):
        assert os.path.exists(path), "Trying to load a vocabulary from a non-existing file {}".format(path)
        with open(path, 'r') as infile:
            all_data = json.load(infile)
            sos_token = all_data["sos_token"]
            eos_token = all_data["eos_token"]
            pad_token = all_data["pad_token"]
            vocab = cls(sos_token=sos_token, eos_token=eos_token, pad_token=pad_token)
            vocab._idx_to_word = all_data["idx_to_word"]
            vocab._word_to_idx = defaultdict(int)
            for word, idx in all_data["word_to_idx"].items():
                vocab._word_to_idx[word] = idx
            vocab._word_frequencies = Counter(all_data["word_frequencies"])
        return vocab

    def to_dict(self) -> dict:
        return {
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "idx_to_word": self._idx_to_word,
            "word_to_idx": self._word_to_idx,
            "word_frequencies": self._word_frequencies
        }

    def save(self, path: str) -> str:
        with open(path, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)
        return path


class GroundedScanEnvironment(gym.Env):
    """
    Loads a GroundedScan RL instance from a specified location.
    """

    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train", input_vocabulary_file="",
                 target_vocabulary_file="", generate_vocabulary=False, seed=42):
        super(GroundedScanEnvironment, self).__init__()
        assert os.path.exists(path_to_data), "Trying to read a gSCAN dataset from a non-existing file {}.".format(
            path_to_data)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if not generate_vocabulary:
            assert os.path.exists(os.path.join(save_directory, input_vocabulary_file)) and os.path.exists(
                os.path.join(save_directory, target_vocabulary_file)), \
                "Trying to load vocabularies from non-existing files."
        if split == "test" and generate_vocabulary:
            logger.warning("WARNING: generating a vocabulary from the test set.")
        self.dataset = GroundedScan.load_dataset_from_file(path_to_data, save_directory=save_directory, k=k)
        if self.dataset._data_statistics.get("adverb_1"):
            logger.info("Verb-adverb combinations in training set: ")
            for adverb, items in self.dataset._data_statistics["train"]["verb_adverb_combinations"].items():
                logger.info("Verbs for adverb: {}".format(adverb))
                for key, count in items.items():
                    logger.info("   {}: {} occurrences.".format(key, count))
            logger.info("Verb-adverb combinations in dev set: ")
            for adverb, items in self.dataset._data_statistics["dev"]["verb_adverb_combinations"].items():
                logger.info("Verbs for adverb: {}".format(adverb))
                for key, count in items.items():
                    logger.info("   {}: {} occurrences.".format(key, count))
        self.image_dimensions = self.dataset.situation_image_dimension
        self.image_channels = 3
        self.split = split
        self.directory = save_directory
        self._num_examples = len(self.dataset._data_pairs[split])
        self._current_example_counter = 0
        self.current_language_observation = None
        self.grid_dimensions = self.dataset.situation_grid_dimensions
        self.progress_reward = False

        # Keeping track of data.
        self._examples = np.array([])
        self._input_lengths = np.array([])
        self._target_lengths = np.array([])
        if generate_vocabulary:
            logger.info("Generating vocabularies...")
            self.input_vocabulary = Vocabulary()
            self.target_vocabulary = Vocabulary()
            self.read_vocabularies()
            logger.info("Done generating vocabularies.")
        else:
            logger.info("Loading vocabularies...")
            self.input_vocabulary = Vocabulary.load(os.path.join(save_directory, input_vocabulary_file))
            self.target_vocabulary = Vocabulary.load(os.path.join(save_directory, target_vocabulary_file))
            logger.info("Done loading vocabularies.")

        self.action_space = spaces.Discrete(self.target_vocabulary_size - 3)  # Don't count <SOS>, <PAD>, <EOS>
        observation_space_grid = spaces.Box(low=0, high=1, shape=(self.grid_dimensions[0],
                                                                  self.grid_dimensions[1],
                                                                  self.grid_dimensions[2]), dtype=np.uint8)
        observation_space_command = spaces.Box(low=0, high=self.input_vocabulary_size - 1,
                                               shape=[1, self.dataset.max_input_length])
        self.observation_space = spaces.Tuple((observation_space_command, observation_space_grid))
        self._random_permutation = [i for i in range(len(self.dataset._data_pairs[self.split]))]

        self.evaluate_on_split = "dev"
        self.num_examples_dev = 100
        self.read_full_training_set = 0
        self.evaluation_examples_idxs = random.sample(range(0,
                                                            len(self.dataset._data_pairs[self.evaluate_on_split])),
                                                      self.num_examples_dev)
        self.shuffle_data()
        self.metadata = {}

    def set_progress_reward(self):
        self.progress_reward = True

    def unset_progress_reward(self):
        self.progress_reward = False

    def read_vocabularies(self) -> {}:
        """
        Loop over all examples in the dataset and add the words in them to the vocabularies.
        """
        logger.info("Populating vocabulary...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split)):
            self.input_vocabulary.add_sentence(example["input_command"])
            self.target_vocabulary.add_sentence(example["target_command"])

    def save_vocabularies(self, input_vocabulary_file: str, target_vocabulary_file: str):
        self.input_vocabulary.save(os.path.join(self.directory, input_vocabulary_file))
        self.target_vocabulary.save(os.path.join(self.directory, target_vocabulary_file))

    def get_vocabulary(self, vocabulary: str) -> Vocabulary:
        if vocabulary == "input":
            vocab = self.input_vocabulary
        elif vocabulary == "target":
            vocab = self.target_vocabulary
        else:
            raise ValueError("Specified unknown vocabulary in sentence_to_array: {}".format(vocabulary))
        return vocab

    def shuffle_data(self) -> {}:
        """
        Reorder the data examples and reorder the lengths of the input and target commands accordingly.
        """
        random_permutation = np.random.permutation(len(self.dataset._data_pairs[self.split]))
        self._random_permutation = random_permutation

    def sentence_to_array(self, sentence: List[str], vocabulary: str, pad_to: int) -> List[int]:
        """
        Convert each string word in a sentence to the corresponding integer from the vocabulary and append
        a start-of-sequence and end-of-sequence token.
        :param sentence: the sentence in words (strings)
        :param vocabulary: whether to use the input or target vocabulary.
        :param pad_to: integer defining until what length to zero-pad the sentence
        :return: the sentence in integers.
        """
        vocab = self.get_vocabulary(vocabulary)
        sentence_array = [vocab.sos_idx]
        for word in sentence:
            sentence_array.append(vocab.word_to_idx(word))
        sentence_array.append(vocab.eos_idx)
        while len(sentence_array) < pad_to:
            sentence_array.append(vocab.pad_idx)
        return sentence_array

    def array_to_sentence(self, sentence_array: List[int], vocabulary: str) -> List[str]:
        """
        Translate each integer in a sentence array to the corresponding word.
        :param sentence_array: array with integers representing words from the vocabulary.
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in words.
        """
        vocab = self.get_vocabulary(vocabulary)
        return [vocab.idx_to_word(word_idx) for word_idx in sentence_array]

    @property
    def max_input_length(self):
        return self.dataset.max_input_length + 2

    def get_evaluation_observations(self) -> List[Tuple[dict, np.ndarray, np.ndarray]]:
        # TODO: change this
        observations = []
        for i in self._random_permutation[:self.num_examples_dev]:
            current_example = self.dataset._data_pairs[self.split][i]
            language_observation, world_observation = self.dataset.initialize_rl_example(current_example)
            language_array = self.sentence_to_array(language_observation, vocabulary="input",
                                                    pad_to=self.dataset.max_input_length + 2)
            observations.append((current_example, language_array, world_observation))
        return observations

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.dataset.max_input_length > 0, "Dataset not properly loaded. Max_input_length = {}.".format(
            self.dataset.max_input_length
        )
        self._current_example_counter = self._current_example_counter % 500  # TODO: change
        example_idx = self._random_permutation[self._current_example_counter]
        current_example = self.dataset._data_pairs[self.split][example_idx]
        # self._current_example_counter = self._current_example_counter % self.num_examples
        language_observation, world_observation = self.dataset.initialize_rl_example(current_example)
        language_array = self.sentence_to_array(language_observation, vocabulary="input",
                                                pad_to=self.dataset.max_input_length + 2)
        self.current_language_observation = language_array
        if self._current_example_counter == 500 - 1:
            self.read_full_training_set += 1
            # self.shuffle_data()
        self._current_example_counter += 1
        return language_array, world_observation

    def render(self, mode='human', close=False):
        pass

    @property
    def percentage_progress(self) -> float:
        return (self._current_example_counter / 500) * 100

    def step(self, action: int) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, dict]:
        if action == self.target_vocabulary.pad_idx or action == self.target_vocabulary.eos_idx \
                or action == self.target_vocabulary.sos_idx:
            logger.warning("Taking <pad>, <sos>, or <eos> step.")
            return (self.current_language_observation, self.dataset.get_current_situation_grid_repr()), \
                0, False, {}
        action_str = self.target_vocabulary.idx_to_word(action)
        new_situation, reward, done = self.dataset.take_step(action_str, progress_reward=self.progress_reward)
        return (self.current_language_observation, new_situation), reward, done, {}

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def input_vocabulary_size(self):
        return self.input_vocabulary.size

    @property
    def target_vocabulary_size(self):
        return self.target_vocabulary.size
