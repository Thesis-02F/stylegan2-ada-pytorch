# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

from torch.autograd import Variable

import torch
import torch.nn as nn

import numpy as np
#----------------------------------------------------------------------------
# from torch import nn
from torchvision import models
from torch.nn import functional as F

import torch.utils.model_zoo as model_zoo

# modified cfg
# TODO: add in a separate file
TRAIN_FLAG = False #True
TEXT_EMBEDDING_DIM = 256
TRAIN_NET_E = './pretrained/image_encoder200.pth'
TRAIN_SMOOTH_GAMMA3 = 10
TRAIN_SMOOTH_LAMBDA = 5

def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)

class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if TRAIN_FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.functional.interpolate(x,size=(299, 299), mode='bilinear', align_corners=False)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)

        return features, cnn_code

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.image_encoder = CNN_ENCODER(TEXT_EMBEDDING_DIM).to(self.device)
        img_encoder_path = TRAIN_NET_E
        state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        self.image_encoder.load_state_dict(state_dict)

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        # *DMSM
        def sent_loss(cnn_code, rnn_code, labels, class_ids,
            batch_size, eps=1e-8, ):
            # ### Mask mis-match samples  ###
            # that come from the same class as the real sample ###
            masks = []
            if class_ids is not None:
                for i in range(batch_size):
                    mask = (class_ids == class_ids[i]).astype(np.uint8)
                    mask[i] = 0
                    masks.append(mask.reshape((1, -1)))
                masks = np.concatenate(masks, 0)
                # masks: batch_size x batch_size
                masks = torch.ByteTensor(masks)
                if self.device:
                    masks = masks.to(self.device)
                masks = masks.bool()
            # --> seq_len x batch_size x nef
            if cnn_code.dim() == 2:
                cnn_code = cnn_code.unsqueeze(0)
                rnn_code = rnn_code.unsqueeze(0)
            # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
            cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
            rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
            # scores* / norm*: seq_len x batch_size x batch_size
            # print( cnn_code.shape, rnn_code.shape )
            scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
            norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
            scores0 = scores0 / norm0.clamp(min=eps) * TRAIN_SMOOTH_GAMMA3
            # --> batch_size x batch_size
            scores0 = scores0.squeeze()
            if class_ids is not None:
                scores0.data.masked_fill_(masks, -float('inf'))
            scores1 = scores0.transpose(0, 1)
            if labels is not None:
                # print(self.device, ':',scores0.device, labels.device)
                loss0 = nn.CrossEntropyLoss()(scores0, labels)
                loss1 = nn.CrossEntropyLoss()(scores1, labels)
            else:
                loss0, loss1 = None, None
            return loss0, loss1

        # DMSM = True
        # if DMSM:

        #     with torch.autograd.profiler.record_function('DMSM_forward'):
        #         gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
        #         #CNN embeddings

        #         #! Change batch size when needed
        #         batch_size = 4
        #         match_labels = Variable(torch.LongTensor(range(batch_size)))
        #         class_ids = None
        #         s_loss0, s_loss1 = sent_loss(cnn_code, gen_c,
        #             match_labels, class_ids, batch_size)
        #         s_loss = (s_loss0 + s_loss1) * TRAIN_SMOOTH_LAMBDA
                
        #         training_stats.report('Loss/G/loss_DMSM', s_loss)
        #     with torch.autograd.profiler.record_function('DMSM_backward'):
        #         s_loss.backward()

        if do_Gmain:
            # Image Encoder
            # image_encoder = CNN_ENCODER(TEXT_EMBEDDING_DIM).to(self.device)
            
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            self.image_encoder.eval()
            self.image_encoder=self.image_encoder.to(self.device)

            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                region_features, cnn_code = self.image_encoder(gen_img)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                #! Change batch size when needed
                batch_size = 16
                match_labels = Variable(torch.LongTensor(range(batch_size))).to(self.device)
                class_ids = None
                s_loss0, s_loss1 = sent_loss(cnn_code, gen_c,
                    match_labels, class_ids, batch_size)
                s_loss = (s_loss0 + s_loss1) * TRAIN_SMOOTH_LAMBDA
                
                training_stats.report('Loss/G/loss', loss_Gmain)
                training_stats.report('Loss/G/DAMSM', s_loss)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain.mean().mul(gain)+s_loss).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
