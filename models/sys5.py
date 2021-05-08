# ==================================================================================================
# Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics
# Author: Jennifer Williams (j.williams@ed.ac.uk)
# All rights reserved.
# ==================================================================================================

import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.dsp import *
import sys
import time
from layers.gradient_reversal import GradientReversal
from layers.temporal_pooling import TemporalPooling_spk
from layers.temporal_pooling import TemporalPooling_phn
from layers.overtone import Overtone
from layers.vector_quant import VectorQuant
from layers.downsampling_encoder import DownsamplingEncoder
from layers.speaker_classifier import SpeakerClassifier
import utils.env as env
import utils.logger as logger
import random
import pytorch_warmup as warmup

class Model(nn.Module) :
    def __init__(self, nspeakers, rnn_dims, fc_dims, global_decoder_cond_dims, upsample_factors, normalize_vq=False,
            noise_x=False, noise_y=False):
        super().__init__()
        self.nspeakers = nspeakers
        self.n_phn_codebooks = 512
        self.vec_len = 128
        self.n_spk_codebooks = 256
        
        self.vq = VectorQuant(1, self.n_phn_codebooks, self.vec_len, normalize=normalize_vq)
        self.vqspk = VectorQuant(1, self.n_spk_codebooks, self.vec_len, normalize=normalize_vq)

        self.noise_x = noise_x
        self.noise_y = noise_y
        encoder_layers_wave = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(128, encoder_layers_wave)
        encoder_layers_spk = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.encoder_spk = DownsamplingEncoder(128, encoder_layers_spk)
        
        self.tap_spk = TemporalPooling_spk(128)
        self.tap_phone = TemporalPooling_phn(128)

        self.speaker_classifier1 = SpeakerClassifier(128, self.nspeakers)
        self.speaker_classifier2 = SpeakerClassifier(128, self.nspeakers)
        self.grl = GradientReversal(scale=1.0)

        self.frame_advantage = 15
        self.num_params()

        self.overtone = Overtone(rnn_dims, fc_dims, 128, 128)
        self.left = self.pad_left()
        self.win = 16 * self.total_scale()
        self.right = self.pad_right()



        
    def forward(self, x, samples):
        continuous_phn = self.encoder(samples)
        continuous_spk = self.encoder_spk(samples)
        # gradient reversal layer here
        phn_GRL = self.grl(continuous_phn)
        # embeddings here        
        tap_output_phn = self.tap_phone(phn_GRL)
        tap_output_spk = self.tap_spk(continuous_spk)
        emb_spk = tap_output_spk.permute(0, 2, 1).squeeze(2)
        emb_phn = tap_output_phn.permute(0, 2, 1).squeeze(2)
        # speaker classifiers here
        spk_predict_spk = self.speaker_classifier1(emb_spk)
        spk_predict_phn = self.speaker_classifier2(emb_phn)
        discrete, vq_pen, encoder_pen, entropy, _ = self.vq(continuous_phn.unsqueeze(2))
        discrete_spk, vq_pen_spk, encoder_pen_spk, entropy_spk, _ = self.vqspk(tap_output_spk.unsqueeze(2))        
        code_x = discrete.squeeze(2)
        discrete_spk = discrete_spk.squeeze(2)
        global_spk = discrete_spk.squeeze(1)
        return self.overtone(x, code_x, global_spk), vq_pen.mean(), encoder_pen.mean(), entropy, vq_pen_spk.mean(), encoder_pen_spk.mean(), entropy_spk, spk_predict_spk, spk_predict_phn

    def after_update(self):
        self.overtone.after_update()
        self.vq.after_update()

    def generate(self, x, A, deterministic=False, use_half=False, verbose=False):
        self.eval()
        with torch.no_grad() :
            continuous = self.encoder(x)
            continuous_spk = self.encoder_spk(A)
            tap_spk = self.tap_spk(continuous_spk)
            discrete, vq_pen, encoder_pen, entropy, _ = self.vq(continuous.unsqueeze(2))
            discrete_spk, vq_pen_spk, encoder_pen_spk, entropy_spk, _ = self.vqspk(tap_spk.unsqueeze(2))
            code_x = discrete.squeeze(2)
            discrete_spk = discrete_spk.squeeze(2)
            global_spk = discrete_spk.squeeze(1)
            output = self.overtone.generate(code_x, global_spk)
        return output


    
    def forward_validate(self, x, A, deterministic=False, use_half=False, verbose=False):
        if use_half:
            A = A.half()
        self.eval()
        with torch.no_grad() :
            continuous_phn = self.encoder(A)
            continuous_spk = self.encoder_spk(A)
            tap_output_phn = self.tap_phone(continuous_phn)
            tap_output_spk = self.tap_spk(continuous_spk)
            emb_spk = tap_output_spk.permute(0, 2, 1).squeeze(2)
            emb_phn = tap_output_phn.permute(0, 2, 1).squeeze(2)
            spk_predict_spk = self.speaker_classifier1(emb_spk)
            # gradient reversal layer here
            phn_GRL = self.grl(emb_phn)
            spk_predict_phn = self.speaker_classifier2(phn_GRL)
            discrete, vq_pen, encoder_pen, entropy, _ = self.vq(continuous_phn.unsqueeze(2))
            discrete_spk, vq_pen_spk, encoder_pen_spk, entropy_spk, _ = self.vqspk(tap_output_spk.unsqueeze(2))
        self.train()
        return vq_pen.mean(), encoder_pen.mean(), entropy, vq_pen_spk.mean(), encoder_pen_spk.mean(), entropy_spk, spk_predict_spk, spk_predict_phn


    def load_state_dict(self, dict, strict=True):
        if strict:
            return super().load_state_dict(self.upgrade_state_dict(dict))
        else:
            my_dict = self.state_dict()
            new_dict = {}
            for key, val in dict.items():
                if key not in my_dict:
                    logger.log(f'Ignoring {key} because no such parameter exists')
                elif val.size() != my_dict[key].size():
                    logger.log(f'Ignoring {key} because of size mismatch')
                else:
                    logger.log(f'Loading {key}')
                    new_dict[key] = val
            return super().load_state_dict(new_dict, strict=False)

    def upgrade_state_dict(self, state_dict):
        out_dict = state_dict.copy()
        return out_dict

    def freeze_encoder(self):
        for name, param in self.named_parameters():
            if name.startswith('encoder.') or name.startswith('vq.'):
                logger.log(f'Freezing {name}')
                param.requires_grad = False
            else:
                logger.log(f'Not freezing {name}')

    def pad_left(self):
        return max(self.pad_left_decoder(), self.pad_left_encoder())

    def pad_left_decoder(self):
        return self.overtone.pad()

    def pad_left_encoder(self):
        return self.encoder.pad_left + (self.overtone.cond_pad - self.frame_advantage) * self.encoder.total_scale

    def pad_right(self):
        return self.frame_advantage * self.encoder.total_scale

    def total_scale(self):
        return self.encoder.total_scale

    def tmp_func(self, batch):
        return env.collate_multispeaker_samples(self.left, self.win, self.right, batch)

    def tmp_func2(self, batch):
        return env.collate_multispeaker_samples_forward(self.left, self.win, self.right, batch)

    
    def do_train(self, paths, dataset, optimiser, epochs, batch_size, num_workers, step, train_sampler, device, lr=1e-4,  spk_lr=0.01, valid_index=[], use_half=False, do_clip=False):

        if use_half:
            import apex
            optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
        for p in optimiser.param_groups : p['lr'] = lr
        criterion = nn.NLLLoss().cuda()
        spk_criterion = torch.nn.CrossEntropyLoss().cuda()
        
        k = 0
        saved_k = 0
        pad_left = self.pad_left()
        pad_left_encoder = self.pad_left_encoder()
        pad_left_decoder = self.pad_left_decoder()
        if self.noise_x:
            extra_pad_right = 127
        else:
            extra_pad_right = 0
        pad_right = self.pad_right() + extra_pad_right
        window = 16 * self.total_scale()
        logger.log(f'pad_left={pad_left_encoder}|{pad_left_decoder}, pad_right={pad_right}, total_scale={self.total_scale()}')

        # from haoyu: slow start for the first 10 epochs
        lr_lambda = lambda epoch: min((epoch) / 10 , 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
        ## warmup is useful !!!
        # warmup_scheduler = warmup.UntunedLinearWarmup(optimiser)
        # warmup_scheduler.last_step = -1

        for e in range(epochs) :
            self.left = self.pad_left()
            self.win = 16 * self.total_scale()
            self.right = pad_right

            trn_loader = DataLoader(dataset, collate_fn=self.tmp_func, batch_size=batch_size,
                num_workers=num_workers, shuffle=(train_sampler is None),  sampler=train_sampler, pin_memory=True)

            start = time.time()
            running_loss_c = 0.
            running_loss_f = 0.
            running_loss_vq = 0.
            running_loss_vqc = 0.

            running_loss_vq_spk = 0.
            running_loss_vqc_spk = 0.
            running_entropy_spk = 0.
            
            running_entropy = 0.
            running_max_grad = 0.
            running_max_grad_name = ""

            running_spk_class_ent1 = 0.
            running_spk_class_acc1 = 0.
            running_spk_class_ent2 = 0.
            running_spk_class_acc2 = 0.
            
            max_loss_c = 0.
            max_loss_f = 0.
            max_loss_vq = 0.
            max_loss_vqc = 0.
            max_loss_vqcw = 0.
            max_loss_vq_spk = 0.
            max_loss_vqc_spk = 0.
            max_loss_vqcw_spk = 0.
            min_loss_c = 10000.
            min_loss_f = 10000.
            min_loss_vq = 10000.
            min_loss_vqc = 10000.
            min_loss_vqcw = 10000.
            min_loss_vq_spk = 10000.
            min_loss_vqc_spk = 10000.
            min_loss_vqcw_spk = 10000.

            running_max_grad = 0.
            running_max_grad_name = ""
            running_min_grad = 1.
            running_min_grad_name = ""

            
            iters = len(trn_loader)

            for i, (speaker, wave16) in enumerate(trn_loader) :

                speaker = speaker.cuda(device)
                wave16 = wave16.cuda(device)

                coarse = (wave16 + 2**15) // 256
                fine = (wave16 + 2**15) % 256

                coarse_f = coarse.float() / 127.5 - 1.
                fine_f = fine.float() / 127.5 - 1.
                total_f = (wave16.float() + 0.5) / 32767.5

                if self.noise_y:
                    noisy_f = total_f * (0.02 * torch.randn(total_f.size(0), 1).cuda(device)).exp() + 0.003 * torch.randn_like(total_f)
                else:
                    noisy_f = total_f

                if use_half:
                    coarse_f = coarse_f.half()
                    fine_f = fine_f.half()
                    noisy_f = noisy_f.half()

                x = torch.cat([
                    coarse_f[:, pad_left-pad_left_decoder:-pad_right].unsqueeze(-1),
                    fine_f[:, pad_left-pad_left_decoder:-pad_right].unsqueeze(-1),
                    coarse_f[:, pad_left-pad_left_decoder+1:1-pad_right].unsqueeze(-1),
                    ], dim=2)
                y_coarse = coarse[:, pad_left+1:1-pad_right]
                y_fine = fine[:, pad_left+1:1-pad_right]

                if self.noise_x:
                    # Randomly translate the input to the encoder to encourage
                    # translational invariance
                    total_len = coarse_f.size(1)
                    translated = []
                    for j in range(coarse_f.size(0)):
                        shift = random.randrange(256) - 128
                        translated.append(noisy_f[j, pad_left-pad_left_encoder+shift:total_len-extra_pad_right+shift])
                    translated = torch.stack(translated, dim=0)
                else:
                    translated = noisy_f[:, pad_left-pad_left_encoder:]

                p_cf, vq_pen, encoder_pen, entropy, vq_pen_spk, encoder_pen_spk, entropy_spk, spk_predict1, spk_predict2 = self(x, translated)
                spk_true = torch.max(speaker, 1)[1]
                spk_loss1 = spk_criterion(spk_predict1, spk_true)
                entropy_spk_class1 = spk_loss1
                accuracy_spk_class1 = 100.0 * (torch.argmax(spk_predict1.data, 1) == spk_true.cuda(device)).sum() / float(spk_true.shape[0])  
                spk_loss2 = spk_criterion(spk_predict2, spk_true)
                if spk_loss2 > 10.0:
                    spk_loss2 = 10.0
                entropy_spk_class2 = spk_loss2
                accuracy_spk_class2 = 100.0 * (torch.argmax(spk_predict2.data, 1) == spk_true.cuda(device)).sum() / float(spk_true.shape[0])  
                

                
                p_c, p_f = p_cf
                loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
                loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
                encoder_weight_phn = 0.0001 * min(1, max(0.01, step / 1000 - 1))
                encoder_weight = 0.01 * min(1, max(0.1, step / 1000 - 1))
                weight_spk = 100
                loss = loss_c + loss_f + (vq_pen + encoder_weight_phn * encoder_pen) + ((vq_pen_spk + encoder_weight * encoder_pen_spk) * weight_spk) + spk_loss1 + spk_loss2


                # min/max loss_c
                if loss_c > max_loss_c:
                    max_loss_c = loss_c
                if loss_c < min_loss_c:
                    min_loss_c = loss_c

                # min/max loss_f
                if loss_f > max_loss_f:
                    max_loss_f = loss_f
                if loss_f < min_loss_f:
                    min_loss_f = loss_f
                    
                # min/max vq_pen
                if vq_pen > max_loss_vq:
                    max_loss_vq = vq_pen
                if vq_pen < min_loss_vq:
                    min_loss_vq = vq_pen

                # min/max encoder_pen
                if encoder_pen > max_loss_vqc:
                    max_loss_vqc = encoder_pen
                if encoder_pen < min_loss_vqc:
                    min_loss_vqc = encoder_pen

                # min/max encoder_pen
                enc_weight = encoder_weight * encoder_pen
                if enc_weight > max_loss_vqcw:
                    max_loss_vqcw = enc_weight
                if enc_weight < min_loss_vqcw:
                    min_loss_vqcw = enc_weight

                # min/max vq_pen SPK
                if vq_pen_spk > max_loss_vq_spk:
                    max_loss_vq_spk = vq_pen_spk
                if vq_pen_spk < min_loss_vq_spk:
                    min_loss_vq_spk = vq_pen_spk

                # min/max encoder_pen SPK
                if encoder_pen_spk > max_loss_vqc_spk:
                    max_loss_vqc_spk = encoder_pen_spk
                if encoder_pen_spk < min_loss_vqc_spk:
                    min_loss_vqc_spk = encoder_pen_spk

                # min/max encoder_pen SPK
                enc_weight_spk = encoder_weight * encoder_pen_spk
                if enc_weight_spk > max_loss_vqcw_spk:
                    max_loss_vqcw_spk = enc_weight_spk
                if enc_weight_spk < min_loss_vqcw_spk:
                    min_loss_vqcw_spk = enc_weight_spk

                optimiser.zero_grad()
                if use_half:
                    optimiser.backward(loss)
                    if do_clip:
                        raise RuntimeError("clipping in half precision is not implemented yet")
                else:
                    loss.backward()
                    if do_clip:
                        max_grad = 0
                        max_grad_name = ""
                        for name, param in self.named_parameters():
                            if param.grad is not None:
                                param_max_grad = param.grad.data.abs().max()
                                if param_max_grad > max_grad:
                                    max_grad = param_max_grad
                                    max_grad_name = name
                                if 1000000 < param_max_grad:
                                    logger.log(f'Very large gradient at {name}: {param_max_grad}')
                        if 100 < max_grad:
                            for param in self.parameters():
                                if param.grad is not None:
                                    if 1000000 < max_grad:
                                        param.grad.data.zero_()
                                    else:
                                        param.grad.data.mul_(100 / max_grad)
                        if running_max_grad < max_grad:
                            running_max_grad = max_grad
                            running_max_grad_name = max_grad_name

                        if 100000 < max_grad:
                            torch.save(self.state_dict(), "bad_model.pyt")
                            raise RuntimeError("Aborting due to crazy gradient (model saved to bad_model.pyt)")
                optimiser.step()
                if e==0 and i==0:
                     lr_scheduler.step()
                     print("schedulre!")
                # lr_scheduler.step()
                # warmup_scheduler.dampen()
                running_loss_c += loss_c.item()
                running_loss_f += loss_f.item()
                running_loss_vq += vq_pen.item()
                running_loss_vqc += encoder_pen.item()
                running_entropy += entropy
                running_loss_vq_spk += vq_pen_spk.item()
                running_loss_vqc_spk += encoder_pen_spk.item()
                running_entropy_spk += entropy_spk
                running_spk_class_ent1 += entropy_spk_class1
                running_spk_class_acc1 += accuracy_spk_class1
                running_spk_class_ent2 += entropy_spk_class2
                running_spk_class_acc2 += accuracy_spk_class2

                self.after_update()

                speed = (i + 1) / (time.time() - start)
                avg_loss_c = running_loss_c / (i + 1)
                avg_loss_f = running_loss_f / (i + 1)
                avg_loss_vq = running_loss_vq / (i + 1)
                avg_loss_vqc = running_loss_vqc / (i + 1)
                avg_entropy = running_entropy / (i + 1)
                avg_loss_vq_spk = running_loss_vq_spk / (i + 1)
                avg_loss_vqc_spk = running_loss_vqc_spk / (i + 1)
                avg_entropy_spk = running_entropy_spk / (i + 1)
                avg_spk_class_ent1 = running_spk_class_ent1 / (i + 1)
                avg_spk_class_acc1 = running_spk_class_acc1 / (i + 1)
                avg_spk_class_ent2 = running_spk_class_ent2 / (i + 1)
                avg_spk_class_acc2 = running_spk_class_acc2 / (i + 1)

                step += 1
                k = step // 1000
                logger.status(f'Ep:{e+1}/{epochs} -- Bt:{i+1}/{iters} '
f'-- Lc={avg_loss_c:#.4} -- Lf={avg_loss_f:#.4} -- Lvq={avg_loss_vq:#.4} -- Lvqc={avg_loss_vqc:#.4} -- LvqS={avg_loss_vq_spk:#.4} -- LvqcS={avg_loss_vqc_spk:#.4} '
f'-- EntS={avg_spk_class_ent1:#.4} -- AccS={avg_spk_class_acc1:#.4} -- EntP={avg_spk_class_ent2:#.4} -- AccP={avg_spk_class_acc2:#.4} -- S:{speed:#.4} steps/sec -- Step: {k}k'
f'\nMinEp:{e+1}/{epochs} -- Bt:{i+1}/{iters} '
f'-- Lc={min_loss_c:#.4} -- Lf={min_loss_f:#.4} -- Lvq={min_loss_vq:#.4} -- Lvqc={min_loss_vqc:#.4} -- Lvqcw={min_loss_vqcw:#.4} -- LvqS={min_loss_vq_spk:#.4} -- LvqcS={min_loss_vqc_spk:#.4} -- LvqcwS={min_loss_vqcw_spk:#.4}'
f'\nMaxEp:{e+1}/{epochs} -- Bt:{i+1}/{iters} '
f'-- Lc={max_loss_c:#.4} -- Lf={max_loss_f:#.4} -- Lvq={max_loss_vq:#.4} -- Lvqc={max_loss_vqc:#.4} -- Lvqcw={max_loss_vqcw:#.4} -- LvqS={max_loss_vq_spk:#.4} -- LvqcS={max_loss_vqc_spk:#.4} -- LvqcwS={max_loss_vqcw_spk:#.4}')



            val_loss_c, val_loss_f, val_loss_vq, val_loss_vqc, val_entropy, val_loss_vq_spk, val_loss_vqc_spk, val_entropy_spk, val_loss_spkclass1, val_acc_spkclass1, val_loss_spkclass2, val_acc_spkclass2 = self.validate(paths, step, dataset.path, valid_index, device)
            logger.status(f'Valid: -- Lc={val_loss_c:#.4} -- Lf={val_loss_f:#.4} -- Lvq={val_loss_vq:#.4} -- Lvqc={val_loss_vqc:#.4} -- LvqS={val_loss_vq_spk:#.4} -- LvqcS={val_loss_vqc_spk:#.4} -- Ls={val_loss_spkclass1:#.4} -- As={val_acc_spkclass1:#.4} -- Lp={val_loss_spkclass2:#.4} -- Ap={val_acc_spkclass2:#.4}')


                
                
            os.makedirs(paths.checkpoint_dir, exist_ok=True)
            torch.save(self.state_dict(), paths.model_path())
            np.save(paths.step_path(), step)
            logger.log_current_status()
            logger.log(f' <saved>; w[0][0] = {self.overtone.wavernn.gru.weight_ih_l0[0][0]}')
            if k > saved_k + 5:
                torch.save(self.state_dict(), paths.model_hist_path(step))
                saved_k = k


    def validate(self, paths, step, data_path, test_index, device, deterministic=False, use_half=False, verbose=False):
        k = step // 1000
        os.makedirs(paths.gen_path(), exist_ok=True)
        criterion = nn.NLLLoss().cuda()
        spk_criterion = torch.nn.CrossEntropyLoss().cuda()

        batch_size = 10
        dataset = env.MultispeakerDataset(test_index, data_path)
        test_loader = DataLoader(dataset, collate_fn=self.tmp_func2, batch_size=batch_size, pin_memory=True)
        running_loss_c = 0.
        running_loss_f = 0.
        running_loss_vq = 0.
        running_loss_vqc = 0.
        running_entropy = 0.
        running_loss_vq_spk = 0.
        running_loss_vqc_spk = 0.
        running_entropy_spk = 0.
        running_loss_spkclass1 = 0.
        running_acc_spkclass1 = 0.
        running_loss_spkclass2 = 0.
        running_acc_spkclass2 = 0.
        iters = len(test_loader)        
        pad_left = self.pad_left()
        pad_left_encoder = self.pad_left_encoder()
        pad_left_decoder = self.pad_left_decoder()
        pad_right = self.pad_right()
        for i, (speaker, wave16, path) in enumerate(test_loader) :
            wave_write = wave16
            wave16 = wave16.cuda()
            speaker = speaker.long().cuda()
            total_f = (wave16.float() + 0.5) / 32767.5
            translated = total_f[:, pad_left-pad_left_encoder:]
            inv_idx = torch.arange(translated.size(0)-1, -1, -1).long()
            inv_translated = translated[inv_idx]            
            audio_files = [np.load(p) for p in path]
            n_points = len(audio_files)
            gt = [(x.astype(np.float32) + 0.5) / (2**15 - 0.5) for x in audio_files]
            extended = [np.concatenate([np.zeros(self.pad_left_encoder(), dtype=np.float32), x, np.zeros(self.pad_right(), dtype=np.float32)]) for x in gt]
            maxlen = max([len(x) for x in extended])
            aligned = [torch.cat([torch.FloatTensor(x).cuda(), torch.zeros(maxlen-len(x)).cuda()]) for x in extended]
            samples = torch.stack(aligned, dim=0).cuda()
            vq_pen, encoder_pen, entropy, vq_pen_spk, encoder_pen_spk, entropy_spk, spk_predict1, spk_predict2 = self.forward_validate(samples, translated)
            running_loss_c += 0
            running_loss_f += 0

            running_loss_vq += vq_pen.item()
            running_loss_vqc += encoder_pen.item()
            running_entropy += entropy
            
            running_loss_vq_spk += vq_pen_spk.item()
            running_loss_vqc_spk += encoder_pen_spk.item()
            running_entropy_spk += entropy_spk

            spk_true = torch.max(speaker, 1)[1]
            spk_loss1 = spk_criterion(spk_predict1, spk_true)
            spk_acc1 = 100.0 * (torch.argmax(spk_predict1.data, 1) == spk_true.cuda(device)).sum() / float(spk_true.shape[0])  
            spk_loss2 = spk_criterion(spk_predict2, spk_true)
            spk_acc2 = 100.0 * (torch.argmax(spk_predict2.data, 1) == spk_true.cuda(device)).sum() / float(spk_true.shape[0])  

            running_loss_spkclass1 += spk_loss1
            running_acc_spkclass1 += spk_acc1
            running_loss_spkclass2 += spk_loss2
            running_acc_spkclass2 += spk_acc2
            
        return running_loss_c/iters, running_loss_f/iters, running_loss_vq/iters, running_loss_vqc/iters, running_entropy/iters, running_loss_vq_spk/iters, running_loss_vqc_spk/iters, running_entropy_spk/iters, running_loss_spkclass1/iters, running_acc_spkclass1/iters, running_loss_spkclass2/iters, running_acc_spkclass2/iters




   
