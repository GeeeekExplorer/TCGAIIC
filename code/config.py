train1_path = '/tcdata/gaiic_track3_round1_train_20210228.tsv'
testA_path = '/tcdata/gaiic_track3_round1_testA_20210228.tsv'
testB_path = '/tcdata/gaiic_track3_round1_testB_20210317.tsv'
train2_path = '/tcdata/gaiic_track3_round2_train_20210407.tsv'
all_path = '../user_data/gaiic_track3_round1_all_20210228.tsv'
vocab_path = '../user_data/vocab.txt'
onnx_path = '../user_data/onnx/'
model_path = '../user_data/model/'
pretrain_path = '../user_data/pretrain/'
result_path = '../prediction_result/result.tsv'
hf_path = '../user_data/huggingface/'
max_seq_length = 32
min_freq = 5
folds = 5
epochs = 50
batch_size = 360
eval_size = 50000
learning_rate = 8e-5
vocab_size = 15000
hidden_size = 768
num_hidden_layers = 12
num_attention_heads = 12
intermediate_size = 768 * 4
num_hidden_states = 768
dropout_prob = 0.3
