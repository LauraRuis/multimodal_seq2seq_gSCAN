import argparse
import logging
import os
import torch

from seq2seq.gSCAN_dataset import GroundedScanDataset
from seq2seq.rl_dataset import GroundedScanEnvironment
from seq2seq.model import Model
from seq2seq.train import train
from seq2seq.predict import predict_and_save
from seq2seq.ppo import train_ppo
from seq2seq.visualize_rewards import visualize_rewards

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger("GroundedSCAN_learning")
use_cuda = True if torch.cuda.is_available() else False

if use_cuda:
    logger.info("Using CUDA.")
    logger.info("Cuda version: {}".format(torch.version.cuda))

parser = argparse.ArgumentParser(description="Sequence to sequence models for Grounded SCAN")

# General arguments
parser.add_argument("--mode", type=str, default="rl", help="train, test, predict, rl, or visualize", required=True)
parser.add_argument("--output_directory", type=str, default="output", help="In this directory the models will be "
                                                                           "saved. Will be created if doesn't exist.")
parser.add_argument("--resume_from_file", type=str, default="", help="Full path to previously saved model to load.")

# Data arguments
parser.add_argument("--split", type=str, default="test", help="Which split to get from Grounded Scan.")
parser.add_argument("--data_directory", type=str, default="data/rl_simplified_dev", help="Path to folder with data.")
parser.add_argument("--input_vocab_path", type=str, default="training_input_vocab.txt",
                    help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--target_vocab_path", type=str, default="training_target_vocab.txt",
                    help="Path to file with target vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--generate_vocabularies", dest="generate_vocabularies", default=False, action="store_true",
                    help="Whether to generate vocabularies based on the data.")
parser.add_argument("--load_vocabularies", dest="generate_vocabularies", default=True, action="store_false",
                    help="Whether to use rewardspreviously saved vocabularies.")

# Training and learning arguments
parser.add_argument("--training_batch_size", type=int, default=50)
parser.add_argument("--k", type=int, default=0, help="How many examples from the adverb_1 split to move to train.")
parser.add_argument("--test_batch_size", type=int, default=1, help="Currently only 1 supported due to decoder.")
parser.add_argument("--max_training_examples", type=int, default=None, help="If None all are used.")
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--lr_decay_steps', type=float, default=20000)
parser.add_argument("--adam_beta_1", type=float, default=0.9)
parser.add_argument("--adam_beta_2", type=float, default=0.999)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--evaluate_every", type=int, default=1000, help="How often to evaluate the model by decoding the "
                                                                     "test set (without teacher forcing).")
parser.add_argument("--max_training_iterations", type=int, default=100000)
parser.add_argument("--weight_target_loss", type=float, default=0.3, help="Only used if --auxiliary_task set.")

# Testing and predicting arguments
parser.add_argument("--max_testing_examples", type=int, default=None)
parser.add_argument("--splits", type=str, default="test", help="comma-separated list of splits to predict for.")
parser.add_argument("--max_decoding_steps", type=int, default=30, help="After 30 decoding steps, the decoding process "
                                                                       "is stopped regardless of whether an EOS token "
                                                                       "was generated.")
parser.add_argument("--output_file_name", type=str, default="predict.json")

# Situation Encoder arguments
parser.add_argument("--simple_situation_representation", dest="simple_situation_representation", default=True,
                    action="store_true", help="Represent the situation with 1 vector per grid cell. "
                                              "For more information, read grounded SCAN documentation.")
parser.add_argument("--image_situation_representation", dest="simple_situation_representation", default=False,
                    action="store_false", help="Represent the situation with the full gridworld RGB image. "
                                               "For more information, read grounded SCAN documentation.")
parser.add_argument("--cnn_hidden_num_channels", type=int, default=50)
parser.add_argument("--cnn_kernel_size", type=int, default=7, help="Size of the largest filter in the world state "
                                                                   "model.")
parser.add_argument("--cnn_dropout_p", type=float, default=0., help="Dropout applied to the output features of the "
                                                                     "world state model.")
parser.add_argument("--auxiliary_task", dest="auxiliary_task", default=False, action="store_true",
                    help="If set to true, the model predicts the target location from the joint attention over the "
                         "input instruction and world state.")
parser.add_argument("--no_auxiliary_task", dest="auxiliary_task", default=True, action="store_false")

# Command Encoder arguments
parser.add_argument("--embedding_dimension", type=int, default=25)
parser.add_argument("--num_encoder_layers", type=int, default=1)
parser.add_argument("--encoder_hidden_size", type=int, default=100)
parser.add_argument("--encoder_dropout_p", type=float, default=0., help="Dropout on instruction embeddings and LSTM.")
parser.add_argument("--encoder_bidirectional", dest="encoder_bidirectional", default=True, action="store_true")
parser.add_argument("--encoder_unidirectional", dest="encoder_bidirectional", default=False, action="store_false")

# Decoder arguments
parser.add_argument("--num_decoder_layers", type=int, default=1)
parser.add_argument("--attention_type", type=str, default='bahdanau', choices=['bahdanau', 'luong'],
                    help="Luong not properly implemented.")
parser.add_argument("--decoder_dropout_p", type=float, default=0., help="Dropout on decoder embedding and LSTM.")
parser.add_argument("--decoder_hidden_size", type=int, default=100)
parser.add_argument("--conditional_attention", dest="conditional_attention", default=True, action="store_true",
                    help="If set to true joint attention over the world state conditioned on the input instruction is"
                         " used.")
parser.add_argument("--no_conditional_attention", dest="conditional_attention", default=False, action="store_false")

# Other arguments
parser.add_argument("--seed", type=int, default=42)

# RL-specific arguments
parser.add_argument("--ppo_log_interval", type=int, default=100, help="After how many timesteps to log progress.")
parser.add_argument("--max_episode_length", type=int, default=40, help="After how many timesteps to cut off an "
                                                                       "episode.")
parser.add_argument("--timesteps_per_epoch", type=int, default=2048, help="How many times to interact with the "
                                                                          "environment per epoch.")
parser.add_argument("--num_epochs", type=int, default=10, help="How many epochs to train the policy.")
parser.add_argument("--pi_lr", type=float, default=0.0003, help="Learning rate for policy gradient updates.")
parser.add_argument("--vf_lr", type=float, default=0.001, help="Learning rate for policy gradient updates.")
parser.add_argument("--train_pi_iters", type=int, default=80, help="How many gradient updates per policy update.")
parser.add_argument("--train_v_iters", type=int, default=80, help="How many gradient updates per value f update.")
parser.add_argument("--ppo_log_every", type=int, default=40, help="How often per policy update to log progress.")
parser.add_argument("--visualize_trajectory_every", type=int, default=1000,
                    help="Every how many timesteps to visualize the current trajectory.")
parser.add_argument("--target_kl", type=float, default=0.01, help="Roughly what KL divergence we think is appropriate "
                                                                  "between new and old policies after an update. "
                                                                  "This will get used for early stopping."
                                                                  " (Usually small, 0.01 or 0.05.)")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor. (Always between 0 and 1.)")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="Lambda for GAE-Lambda. "
                                                                   "(Always between 0 and 1, close to 1.)")
parser.add_argument("--ent_coef", type=float, default=0.0, help="")
parser.add_argument("--clip_ratio", type=float, default=0.2, help="")
parser.add_argument("--vf_coef", type=float, default=0.5, help="")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="")
parser.add_argument("--hidden_size", type=int, default=100)
parser.add_argument("--total_timesteps", type=int, default=25000)
parser.add_argument("--evaluate_every_epoch", type=int, default=20, help="How often to evaluate the current policy"
                                                                         " greedily.")
parser.add_argument("--rewards_file", type=str, default="", help="File with reward on each line.")


def main(flags):
    for argument, value in flags.items():
        logger.info("{}: {}".format(argument, value))

    if not os.path.exists(flags["output_directory"]):
        os.mkdir(os.path.join(os.getcwd(), flags["output_directory"]))

    if not flags["simple_situation_representation"]:
        raise NotImplementedError("Full RGB input image not implemented. Implement or set "
                                  "--simple_situation_representation")
    # Some checks on the flags
    if flags["generate_vocabularies"]:
        assert flags["input_vocab_path"] and flags["target_vocab_path"], "Please specify paths to vocabularies to save."

    if flags["test_batch_size"] > 1:
        raise NotImplementedError("Test batch size larger than 1 not implemented.")

    data_path = os.path.join(flags["data_directory"], "dataset.txt")
    if flags["mode"] == "train":
        train(data_path=data_path, **flags)
    elif flags["mode"] == "test":
        assert os.path.exists(os.path.join(flags["data_directory"], flags["input_vocab_path"])) and os.path.exists(
            os.path.join(flags["data_directory"], flags["target_vocab_path"])), \
            "No vocabs found at {} and {}".format(flags["input_vocab_path"], flags["target_vocab_path"])
        splits = flags["splits"].split(",")
        for split in splits:
            logger.info("Loading {} dataset split...".format(split))
            test_set = GroundedScanDataset(data_path, flags["data_directory"], split=split,
                                           input_vocabulary_file=flags["input_vocab_path"],
                                           target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False,
                                           k=flags["k"])
            test_set.read_dataset(max_examples=None,
                                  simple_situation_representation=flags["simple_situation_representation"])
            logger.info("Done Loading {} dataset split.".format(flags["split"]))
            logger.info("  Loaded {} examples.".format(test_set.num_examples))
            logger.info("  Input vocabulary size: {}".format(test_set.input_vocabulary_size))
            logger.info("  Most common input words: {}".format(test_set.input_vocabulary.most_common(5)))
            logger.info("  Output vocabulary size: {}".format(test_set.target_vocabulary_size))
            logger.info("  Most common target words: {}".format(test_set.target_vocabulary.most_common(5)))

            model = Model(input_vocabulary_size=test_set.input_vocabulary_size,
                          target_vocabulary_size=test_set.target_vocabulary_size,
                          num_cnn_channels=test_set.image_channels,
                          input_padding_idx=test_set.input_vocabulary.pad_idx,
                          target_pad_idx=test_set.target_vocabulary.pad_idx,
                          target_eos_idx=test_set.target_vocabulary.eos_idx,
                          **flags)
            model = model.cuda() if use_cuda else model

            # Load model and vocabularies if resuming.
            assert os.path.isfile(flags["resume_from_file"]), "No checkpoint found at {}".format(flags["resume_from_file"])
            logger.info("Loading checkpoint from file at '{}'".format(flags["resume_from_file"]))
            model.load_model(flags["resume_from_file"])
            start_iteration = model.trained_iterations
            logger.info("Loaded checkpoint '{}' (iter {})".format(flags["resume_from_file"], start_iteration))
            output_file_name = "_".join([split, flags["output_file_name"]])
            output_file_path = os.path.join(flags["output_directory"], output_file_name)
            output_file = predict_and_save(dataset=test_set, model=model, output_file_path=output_file_path, **flags)
            logger.info("Saved predictions to {}".format(output_file))
    elif flags["mode"] == "predict":
        raise NotImplementedError()
    elif flags["mode"] == "rl":
        logger.info("Loading Training set...")
        training_env = GroundedScanEnvironment(data_path, flags["data_directory"],
                                               split="train",
                                               seed=flags["seed"],
                                               input_vocabulary_file=flags["input_vocab_path"],
                                               target_vocabulary_file=flags["target_vocab_path"],
                                               generate_vocabulary=flags["generate_vocabularies"], k=1)
        if flags["generate_vocabularies"]:
            training_env.save_vocabularies(flags["input_vocab_path"], flags["target_vocab_path"])
            logger.info(
                "Saved vocabularies to {} for input and {} for target.".format(flags["input_vocab_path"],
                                                                               flags["target_vocab_path"]))
        train_ppo(training_env=training_env, **flags)
    elif flags["mode"] == "visualize":
        visualize_rewards(flags["rewards_file"])
    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(flags["mode"]))


if __name__ == "__main__":
    input_flags = vars(parser.parse_args())
    main(flags=input_flags)
