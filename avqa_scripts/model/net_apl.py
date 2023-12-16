import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from loss import AVContrastive_loss_100 as AVContrastive_loss
from fusion import AttFlat
from transformer_encoder import SAEncoder, CXEncoder


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):
        
        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)
        
    def forward(self, question, qsn_lengths):
        qst_vec = self.word2vec(question)
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)

        self.lstm.flatten_parameters()
        packed = pack_padded_sequence(qst_vec, qsn_lengths.cpu(), batch_first=False, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output)

        qst_feature = torch.cat((hidden, cell), 2)
        qst_feature = qst_feature.transpose(0, 1)
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)

        return output, qst_feature



class APL(nn.Module):
    def __init__(
            self,
            vocab_size=93,
            audio_input_dim=128,
            object_input_dim=256,
            hidden_size=512,
            answer_vocab_size=42,
            q_max_len=14,
            dropout_p1=0.1,
            dropout_p2=0.1,
            sa_encoder_layers_num=1,
            sa_nhead=1,
            sa_d_model=512,
            sa_dim_feedforward=2048,
            cx_encoder_layers_num=1,
            cx_nhead=4,
            cx_d_model=512,
            cx_dim_feedforward=2048):
        
        super().__init__()

        self.q_max_len = q_max_len
        self.hidden_size = hidden_size

        self.sentence_encoder = QstEncoder(qst_vocab_size=vocab_size, word_embed_size=512, embed_size=512, num_layers=1, hidden_size=512)

        self.fc_audio = nn.Linear(audio_input_dim, hidden_size)
        self.fc_object = nn.Linear(object_input_dim + 4, hidden_size)

        self.sa_a_pos_embed = nn.Embedding(10, hidden_size)
        self.sa_audio_encoder = SAEncoder(d_model=sa_d_model, nhead=sa_nhead, num_encoder_layers=sa_encoder_layers_num, dim_feedforward=sa_dim_feedforward, dropout=dropout_p1)

        self.aqq_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num, dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)
        self.oqq_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num, dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)

        self.qaa_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num, dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)
        self.qoo_encoder = CXEncoder(d_model=cx_d_model, nhead=cx_nhead, num_encoder_layers=cx_encoder_layers_num, dim_feedforward=cx_dim_feedforward, dropout=dropout_p1)

        self.wei_from_q = nn.Linear(hidden_size, 2)
        self.attflat = AttFlat(hidden_size, hidden_size, 1, answer_vocab_size, dropout_r=dropout_p2)


    def forward(self, audio_feature, object_feature, bbox_feature, question, ques_len):
        # expected sentence_inputs is of shape (batch_size, sentence_len, 1)
        # expected video_inputs is of shape (batch_size, frame_num, video_feature)
        out, nce_loss_AO, nce_loss_VO = self.model_block(audio_feature, object_feature, bbox_feature, question, ques_len)

        return out, nce_loss_AO, nce_loss_VO
        

    def model_block(self, audio_fea, object_fea, bbox_fea, question, ques_len):
        ######################################################################################
        # Feature preparation
        ######################################################################################
        #* question
        word_fea, sentence_fea = self.sentence_encoder(question, ques_len)
        q_mask = self.make_mask(question, ques_len)
        q_temp = sentence_fea.unsqueeze(1)
        q_repeat = q_temp.repeat(1, 10, 1)

        # *object and bbox
        B, T, OBJECT_NUM, _ = object_fea.shape
        object_fea = object_fea.view(B, T*OBJECT_NUM, -1)
        
        bbox_fea[:,:,:,0] /= bbox_fea [:,:,:,-2]
        bbox_fea[:,:,:,1] /= bbox_fea [:,:,:,-1]
        bbox_fea[:,:,:,2] /= bbox_fea [:,:,:,-2]
        bbox_fea[:,:,:,3] /= bbox_fea [:,:,:,-1]
        
        bbox_xyxy = bbox_fea[:,:,:,0:4]
        bbox_fea = bbox_xyxy.view(B, T*OBJECT_NUM, -1)
        object_box_concat = torch.cat([object_fea, bbox_fea], dim=-1)
        object_fea = self.fc_object(object_box_concat)
        object_fea = object_fea.permute(1, 0, 2)

        #* audio 
        audio_fea = self.fc_audio(audio_fea)
        audio_fea = audio_fea.permute(1, 0, 2)
        sa_a_pos = self.sa_a_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)
        sa_audio_fea, sa_head_audio_fea_list, sa_head_audio_attn_list  = self.sa_audio_encoder(audio_fea, attn_mask=None, key_padding_mask=None, pos_embed=sa_a_pos) # [T, bs, 512]


        ######################################################################################
        # QCD: question modility as 'key' and 'value'
        ######################################################################################
        question_mask = self.gene_question_as_key_pad_mask(word_fea.permute(1,0,2), ques_len)
        question_mask = question_mask.to('cuda')
        
        cx_a_fea, _, _ = self.aqq_encoder(sa_audio_fea, word_fea, attn_mask=None, key_padding_mask=question_mask, q_pos_embed=None, k_pos_embed=None) # [T, bs, 512]
        cx_o_fea, _, _ = self.oqq_encoder(object_fea, word_fea, attn_mask=None, key_padding_mask=question_mask, q_pos_embed=None, k_pos_embed=None) # [T*N, bs, 512]

        ######################################################################################
        # object-aware adaptive contrastive losses.
        ######################################################################################
        nce_loss_OQ = AVContrastive_loss(cx_o_fea.permute(1,0,2), q_repeat)
        nce_loss_OA = AVContrastive_loss(cx_o_fea.permute(1,0,2), cx_a_fea.permute(1,0,2))
        
        ######################################################################################
        # MCC: auido and viusal modality as 'key' and 'value'
        ######################################################################################
        cx_a_fea2, _, _  = self.qaa_encoder(word_fea, cx_a_fea, attn_mask=None, key_padding_mask=None, q_pos_embed=None, k_pos_embed=None) # [cur_max_lenth, bs, 512]
        cx_o_fea2, _, _  = self.qoo_encoder(word_fea, cx_o_fea, attn_mask=None, key_padding_mask=None, q_pos_embed=None, k_pos_embed=None) # [cur_max_lenth, bs, 512]
        cx_a_fea2 = cx_a_fea2.permute(1, 0, 2)
        cx_o_fea2 = cx_o_fea2.permute(1, 0, 2)

        ######################################################################################
        # modality-aware fusion
        ######################################################################################
        modality_wei = self.wei_from_q(sentence_fea)
        modality_wei = torch.softmax(modality_wei, dim=-1)
        cx_a_fea2 = cx_a_fea2 * modality_wei[:, 0].unsqueeze(-1).unsqueeze(-1)
        cx_o_fea2 = cx_o_fea2 * modality_wei[:, 1].unsqueeze(-1).unsqueeze(-1)
        cx_fused_fea = cx_a_fea2 + cx_o_fea2
    
        fusion_out = self.attflat(cx_fused_fea, q_mask)
        return fusion_out, nce_loss_OQ, nce_loss_OA


    def gene_question_as_key_pad_mask(self, q_fea, seq_length):
        mask = torch.ones(q_fea.shape[:2])
        for i, l in enumerate(seq_length):
            mask[i][l:] = 0
        mask = mask.to(torch.bool) 
        mask = ~mask
        return mask

    def make_mask(self, seq, seq_length):
        mask = torch.ones(seq.shape[0], max(seq_length)).cuda()
        
        for i, l in enumerate(seq_length):
            mask[i][l:] = 0
        mask = Variable(mask)
        mask = mask.to(torch.float)
        return mask