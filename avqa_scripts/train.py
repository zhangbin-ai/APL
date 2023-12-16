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
import torch.optim as optim

from configs.args import parser
import os
from pathlib import Path


def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_qa = 0
    correct_qa = 0
    for batch_idx, sample in enumerate(train_loader):

        audio_feature,video_feature, bbox, target, question, ques_len, video_id, question_id = \
            sample['audio_feature'].to('cuda'), sample['video_feature'].to('cuda'), sample['bbox'].to('cuda'), \
            sample['label'].to('cuda'), sample['question'].to('cuda'), sample['ques_len'].to('cuda'), sample['video_id'], sample['question_id']

        optimizer.zero_grad()
        out_qa, nce_loss_OQ, nce_loss_OA = model(audio_feature, video_feature, bbox, question, ques_len)

        loss_qa = criterion(out_qa, target)
        loss = loss_qa + args.loss_oq_wei * nce_loss_OQ + args.loss_oa_wei * nce_loss_OA
        
        loss.backward()
        optimizer.step()


        pred_index, predicted = torch.max(out_qa, 1)
        correct_qa += (predicted == target).sum().item()
        total_qa += out_qa.size(0)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t total Loss: {:.4f}  |  ans_loss:{:.6f}  OQ-loss:{:.4f}  OA-loss:{:.4f}'.format(
                epoch, batch_idx * len(audio_feature), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), loss_qa.item(), nce_loss_OQ.item(), nce_loss_OA.item()), flush=True)
            
    return correct_qa, total_qa, 100 * correct_qa / total_qa


def eval(model, val_loader):
    model.eval()
    
    total_qa = 0
    correct_qa = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):

            audio_feature,video_feature, bbox, target, question, ques_len, video_id, question_id = \
                sample['audio_feature'].to('cuda'), sample['video_feature'].to('cuda'), sample['bbox'].to('cuda'), \
                sample['label'].to('cuda'), sample['question'].to('cuda'), sample['ques_len'].to('cuda'), sample['video_id'], sample['question_id']

            preds_qa, _, _ = model(audio_feature, video_feature, bbox,question, ques_len)
            _, predicted = torch.max(preds_qa, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()

    print('Accuracy val_set qa: %.2f %%' % (100 * correct_qa / total_qa), flush=True)

    return 100 * correct_qa / total_qa


def main():
    
    args = parser.parse_args()
    print(format("main.py path", '<25'), Path(__file__).resolve())

    for arg in vars(args):
        print(format(arg, '<25'), format(str(getattr(args, arg)), '<'))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.model == 'APL':

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

        model=nn.DataParallel(model)
        model=model.to('cuda')
        
    else:
        raise ('not recognized')


    if args.mode == 'train':
        train_dataset = AVQA_dataset(label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                    transform=transforms.Compose([ToTensor()]), mode_flag='train')
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
        


        val_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                   transform=transforms.Compose([ToTensor()]), mode_flag='val')
        
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)



        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.steplr_step, gamma=args.steplr_gamma)
        
        criterion = nn.CrossEntropyLoss()
        
        
        val_best = 0
        
        for epoch in range(1, args.epochs + 1):
            print(f"\nthe {epoch}-th learning rate is {optimizer.param_groups[0]['lr']}")

            #########################################################################################################
            # !!! train
            #########################################################################################################
            correct_qa, total_qa, train_acc = train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            print('Accuracy train_set qa: %.2f %%' % (100 * correct_qa / total_qa), flush=True)
            scheduler.step(epoch)
            
            
            #########################################################################################################
            # !!! val
            #########################################################################################################
            val_acc = eval(model, val_loader)
            
            if val_acc >= val_best:

                val_best = val_acc
                save_model_folder = Path(args.model_save_dir, args.checkpoint_file)
                save_model_path = Path(save_model_folder, f"model_{epoch}.pt")
                
                # save model file folder
                if not save_model_folder.exists():
                    save_model_folder.mkdir()
                
                if args.save_model_flag == 'True':
                    torch.save(model.state_dict(), str(save_model_path))
                    print(">>>save model path:", save_model_path)
                else:
                    print(">>>not save model.")



if __name__ == '__main__':

    main()