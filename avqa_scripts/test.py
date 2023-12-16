import sys
sys.path.append("../../APL_DETR")

import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from avqa_scripts.dataloader import AVQA_dataset, ToTensor
from avqa_scripts.model.net_apl import APL

from configs.args import parser
import ast
import os
import json


def test(model, test_loader, test_json_file):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open(test_json_file, 'r'))

    # useing index of question
    questionid_to_samples = {}
    for sample in samples:
        ques_id = sample['question_id']
        if ques_id not in questionid_to_samples.keys():
            questionid_to_samples[ques_id] = sample
        else:
            print("question_id_duplicated:", ques_id)

    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):

            audio_feature,video_feature, bbox, target, question, ques_len, video_id, question_id = \
                sample['audio_feature'].to('cuda'), sample['video_feature'].to('cuda'), sample['bbox'].to('cuda'), \
                sample['label'].to('cuda'), sample['question'].to('cuda'), sample['ques_len'].to('cuda'), sample['video_id'], sample['question_id']

            preds_qa, _, _ = model(audio_feature, video_feature, bbox, question, ques_len)

            preds = preds_qa
            _, predicted = torch.max(preds.data, 1)
            total += preds.size(0)
            correct += (predicted == target).sum().item()

            question_id = question_id.numpy().tolist()

            for index, ques_id in enumerate(question_id):
                x = questionid_to_samples[ques_id]
                type =ast.literal_eval(x['type'])

                if type[0] == 'Audio':
                    if type[1] == 'Counting':
                        A_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Comparative':
                        A_cmp.append((predicted[index] == target[index]).sum().item())
                elif type[0] == 'Visual':
                    if type[1] == 'Counting':
                        V_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Location':
                        V_loc.append((predicted[index] == target[index]).sum().item())
                elif type[0] == 'Audio-Visual':
                    if type[1] == 'Existential':
                        AV_ext.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Counting':
                        AV_count.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Location':
                        AV_loc.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Comparative':
                        AV_cmp.append((predicted[index] == target[index]).sum().item())
                    elif type[1] == 'Temporal':
                        AV_temp.append((predicted[index] == target[index]).sum().item())
    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))
    
    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))
    
    return 100 * correct / total


def main():
    args = parser.parse_args()

    test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir, \
                    transform=transforms.Compose([ToTensor()]), mode_flag='test')
        
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=6, pin_memory=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = APL(
            vocab_size = args.vocab_size,
            audio_input_dim = args.audio_input_dim,
            object_input_dim = args.object_input_dim,
            hidden_size = args.hidden_size,
            answer_vocab_size = args.answer_vocab_size,
            q_max_len = args.q_max_len,
            dropout_p1=args.dropout_p1,
            dropout_p2=args.dropout_p2,
            sa_encoder_layers_num = args.sa_encoder_layers_num,
            sa_nhead = args.sa_nhead,
            sa_d_model = args.sa_d_model,
            sa_dim_feedforward = args.sa_dim_feedforward,
            cx_encoder_layers_num = args.cx_encoder_layers_num,
            cx_nhead = args.cx_nhead,
            cx_d_model = args.cx_d_model,
            cx_dim_feedforward = args.cx_dim_feedforward
    )

    model = nn.DataParallel(model)
    model = model.to('cuda')
    
    model.load_state_dict(torch.load(args.model_save_dir + "20.pt"))
    test(model, test_loader, args.label_test)



if __name__ == '__main__':

    main()