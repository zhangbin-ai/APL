import os
import argparse


parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Question Answering')

dataset_path_prefix = "/home/data/music"

#* dataloader.
parser.add_argument( "--audio_dir", type=str, default=os.path.join(dataset_path_prefix, 'vggish'), help="audio dir")
parser.add_argument( "--video_dir", type=str, default=os.path.join(dataset_path_prefix, 'detr_bbox'), help="video dir")

#* train setting
parser.add_argument( "--label_train", type=str, default="../../APL_DETR/avqa_data/avqa-train.json", 
                    help="train csv file")
parser.add_argument( "--label_val", type=str, default="../../APL_DETR/avqa_data/avqa-val.json", 
                    help="val csv file")
parser.add_argument( "--label_test", type=str, default="../../APL_DETR/avqa_data/avqa-test.json", 
                    help="test csv file")

parser.add_argument( '--batch_size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 16)')
parser.add_argument( '--test_batch_size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 16)')
parser.add_argument( '--epochs', type=int, default=30, metavar='N', 
                    help='number of epochs to train (default: 60)')
parser.add_argument( "--model", type=str, default='APL', 
                    help="with model to use")


# parser.add_argument( "--mode", type=str, default='train', help="with mode to use")
parser.add_argument( "--save_model_flag", type=str, default='True', 
                    help="flag as save model")
parser.add_argument( '--seed', type=int, default=42, metavar='S', 
                    help='random seed (default: 1)')
parser.add_argument( '--log-interval', type=int, default=50, metavar='N', 
                    help='how many batches to wait before logging training status')
parser.add_argument( '--gpu', type=str, default='0', 
                    help='gpu device number')


# learning rate 
parser.add_argument( '--lr', type=float, default=1e-4, metavar='LR', 
                    help='learning rate')
parser.add_argument( '--loss_oq_wei', type=float, default=0.3, 
                    help='OQ contrastive loss weight')
parser.add_argument( '--loss_oa_wei', type=float, default=0.3, 
                    help='OA contrastive loss weight')
parser.add_argument( '--steplr_step', type=int, default=8, 
                    help='after x steps it goes down')
parser.add_argument( '--steplr_gamma', type=float, default=0.1, 
                    help='after x steps it goes down rate')

# model hyper parameter
parser.add_argument( '--vocab_size', type=int, default=93, 
                    help='question words number')
parser.add_argument( '--audio_input_dim', type=int, default=128, 
                    help='preprocessed audio feature dimensions')
parser.add_argument( '--visual_input_dim', type=int, default=512, 
                    help='preprocessed visual feature dimensions')
parser.add_argument( '--object_input_dim', type=int, default=256, 
                    help='preprocessed obejct feature dimensions')
parser.add_argument( '--hidden_size', type=int, default=512, 
                    help='Hidden layer dimension')
parser.add_argument( '--dropout_p1', type=float, default=0.1, 
                    help='dropout probability of attention')
parser.add_argument( '--dropout_p2', type=float, default=0.1,
                    help='dropout probability of fusion')
parser.add_argument( '--answer_vocab_size', type=int, default=42, 
                    help='answer words number')
parser.add_argument( '--q_max_len', type=int, default=14, 
                    help='max words number of questions')
parser.add_argument( '--sa_encoder_layers_num', type=int, default=1, 
                    help='self-attention layers')
parser.add_argument( '--sa_nhead', type=int, default=1, 
                    help='self-attention heads')
parser.add_argument( '--sa_d_model', type=int, default=512, 
                    help='self-attention output dimensions')
parser.add_argument( '--sa_dim_feedforward', type=int, default=2048, 
                    help='self-attention hidden dimensions')
parser.add_argument( '--cx_encoder_layers_num', type=int, default=1, 
                    help='cross-attention layers')
parser.add_argument( '--cx_nhead', type=int, default=4, 
                    help='cross-attention heads')
parser.add_argument( '--cx_d_model', type=int, default=512, 
                    help='cross-attention output dimensions')
parser.add_argument( '--cx_dim_feedforward', type=int, default=2048, 
                    help='cross-attention hidden dimensions')


#* save path
parser.add_argument( "--model_save_dir", type=str, default='check_models', 
                    help="model save dir")
parser.add_argument( "--checkpoint_file", type=str, default='err_dir', 
                    help="model name")