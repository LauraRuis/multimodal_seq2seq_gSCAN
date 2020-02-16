#!/bin/bash
# Compositional Generalization Experiments
# Baseline
# Training
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=adverb_k_1_run_3 --training_batch_size=200 --max_training_iterations=200000 --seed=126 &> adverb_k_1_run_3/adverb_run.txt
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=adverb_k_1_run_4 --training_batch_size=200 --max_training_iterations=200000 --seed=126 &> adverb_k_1_run_4/adverb_run.txt
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=adverb_k_1_run_5 --training_batch_size=200 --max_training_iterations=200000 --seed=126 &> adverb_k_1_run_5/adverb_run.txt

# Testing
python3.7 -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=adverb_k_1_run_3 --resume_from_file=adverb_k_1_run_3/model_best.pth.tar --splits=test,dev,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=adverb_k_1_run_3.json --max_decoding_steps=120 &> adverb_k_1_run_3/test.txt
python3.7 -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=adverb_k_1_run_4 --resume_from_file=adverb_k_1_run_4/model_best.pth.tar --splits=test,dev,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=adverb_k_1_run_4.json --max_decoding_steps=120 &> adverb_k_1_run_4/test.txt
python3.7 -m seq2seq --mode=test --data_directory=data/compositional_splits --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=adverb_k_1_run_5 --resume_from_file=adverb_k_1_run_5/model_best.pth.tar --splits=test,dev,visual,visual_easier,situational_1,situational_2,contextual,adverb_1,adverb_2 --output_file_name=adverb_k_1_run_5.json --max_decoding_steps=120 &> adverb_k_1_run_5/test.txt

# GECA
# Exactly the same commands as for the baseline except change the --data_directory to data/GECA, the output_directory, and the seed.

# Target Lengths Experiments
# Training
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --training_batch_size=200 --max_training_iterations=200000 --seed=126 &> target_lengths_run_1/target_lengths_run.txt
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --training_batch_size=200 --max_training_iterations=200000 --seed=126 &> target_lengths_run_2/target_lengths_run.txt
python3.7 -m seq2seq --mode=train --max_decoding_steps=120 --max_testing_examples=2000 --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --training_batch_size=200 --max_training_iterations=200000 --seed=126 &> target_lengths_run_3/target_lengths_run.txt

# Testing
python3.7 -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_1 --resume_from_file=target_lengths_run_1/model_best.pth.tar --splits=test,dev,target_lengths --output_file_name=target_lengths_predict_run_1.json --max_decoding_steps=120 &> target_lengths_run_1/test.txt
python3.7 -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_2 --resume_from_file=target_lengths_run_2/model_best.pth.tar --splits=test,dev,target_lengths --output_file_name=target_lengths_predict_run_2.json --max_decoding_steps=120 &> target_lengths_run_2/test.txt
python3.7 -m seq2seq --mode=test --data_directory=data/target_length_split --attention_type=bahdanau --no_auxiliary_task --conditional_attention --output_directory=target_lengths_run_3 --resume_from_file=target_lengths_run_3/model_best.pth.tar --splits=test,dev,target_lengths --output_file_name=target_lengths_predict_run_3.json --max_decoding_steps=120 &> target_lengths_run_3/test.txt