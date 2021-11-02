#!/usr/bin/env python3

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import *
from transformer4 import *
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import argparse
import codecs
import numpy as np
from torch.optim.lr_scheduler import StepLR

def train(args, train_loader, encoder, decoder, criterion, encoder_optimizer,encoder_lr_scheduler, decoder_optimizer, decoder_lr_scheduler, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    best_bleu4 = 0.  # BLEU-4 score right now
    steps_since_improvement = 0
    final_args = {"emb_dim": args.emb_dim,
                  "attention_dim": args.attention_dim,
                  "decoder_dim": args.decoder_dim,
                  "n_heads": args.n_heads,
                  "dropout": args.dropout,
                  "decoder_mode": args.decoder_mode,
                  "attention_method": args.attention_method,
                  "encoder_layers": args.encoder_layers,
                  "decoder_layers": args.decoder_layers}
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        # print(caps)
        # print(caplens)
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        # imgs: [batch_size, 14, 14, 2048]
        # caps: [batch_size, 52]
        # caplens: [batch_size, 1]
        if args.decoder_mode == 'lstm':
            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
        else:
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        # print(scores.size())
        # print(targets.size())

        # Calculate loss
        loss = criterion(scores, targets)
        # Add doubly stochastic attention regularization
        # Second loss, mentioned in paper "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
        # https://arxiv.org/abs/1502.03044
        # In section 4.2.1 Doubly stochastic attention regularization: We know the weights sum to 1 at a given timestep.
        # But we also encourage the weights at a single pixel p to sum to 1 across all timesteps T.
        # This means we want the model to attend to every pixel over the course of generating the entire sequence.
        # Therefore, we want to minimize the difference between 1 and the sum of a pixel's weights across all timesteps.
        if args.decoder_mode == "lstm_attention":
            loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        elif args.decoder_mode == "transformer" or args.decoder_mode == "transformer_decoder":
            dec_alphas = alphas["dec_enc_attns"]
            alpha_trans_c = args.alpha_c / (args.n_heads * args.decoder_layers)
            for layer in range(args.decoder_layers):  # args.decoder_layers = len(dec_alphas)
                cur_layer_alphas = dec_alphas[layer]  # [batch_size, n_heads, 52, 196]
                for h in range(args.n_heads):
                    cur_head_alpha = cur_layer_alphas[:, h, :, :]
                    loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        decoder_lr_scheduler.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()
            encoder_lr_scheduler.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        if i % args.print_freq == 0:
            # print('TIME: ', time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            print("Epoch: {}/{} step: {}/{} Loss: {} AVG_Loss: {} Top-5 Accuracy: {} Batch_time: {}s".format(epoch+0, args.epochs, i+0, len(train_loader), losses.val, losses.avg, top5accs.val, batch_time.val))


def validate(args, val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: score_dict {'Bleu_1': 0., 'Bleu_2': 0., 'Bleu_3': 0., 'Bleu_4': 0., 'METEOR': 0., 'ROUGE_L': 0., 'CIDEr': 1.}
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)

            if args.decoder_mode == 'lstm':
                scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
            else:
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            if args.decoder_mode == "lstm_attention":
                loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            elif args.decoder_mode == "transformer" or args.decoder_mode =="transformer_decoder":
                dec_alphas = alphas["dec_enc_attns"]
                alpha_trans_c = args.alpha_c / (args.n_heads * args.decoder_layers)
                for layer in range(args.decoder_layers):  # args.decoder_layers = len(dec_alphas)
                    cur_layer_alphas = dec_alphas[layer]  # [batch_size, n_heads, 52, 196]
                    for h in range(args.n_heads):
                        cur_head_alpha = cur_layer_alphas[:, h, :, :]
                        loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()


            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    print('Validation：')
    metrics = get_eval_score(references, hypotheses)

    # print("EVA LOSS: {} TOP-5 Accuracy {} BLEU-1 {} BLEU2 {} BLEU3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format
    #       (losses.avg, top5accs.avg,  metrics["Bleu_1"],  metrics["Bleu_2"],  metrics["Bleu_3"],  metrics["Bleu_4"],
    #        metrics["METEOR"],metrics["ROUGE_L"], metrics["CIDEr"]))
    print('\n')

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image_Captioning')

    # Data parameters
    parser.add_argument('--data_folder', default="D:/LCY/RSICD_captions/data/",help='folder with data files saved by create_input_files.py.')
    parser.add_argument('--data_name', default="RSICD_5_cap_per_img_5_min_word_freq",help='base name shared by data files.')


    # Model parameters
    parser.add_argument('--emb_dim', type=int, default=512, help='dimension of word embeddings.')#300
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers.')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN.')
    parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # FIXME:note to change these
    parser.add_argument('--encoder_mode', default="resnet50", help='which model does encoder use?') # inception_v3 or vgg16 or vgg19 or resnet50 or resnet101 or resnet152
    parser.add_argument('--decoder_mode', default="transformer", help='which model does decoder use?')  # lstm or lstm_attention or transformer or transformer_decoder

    parser.add_argument('--attention_method', default="ByPixel", help='which attention method to use?')  # ByPixel or ByChannel
    parser.add_argument('--encoder_layers', type=int, default=3, help='the number of layers of encoder in Transformer.')
    parser.add_argument('--decoder_layers', type=int, default=3, help='the number of layers of decoder in Transformer.')


    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for (if early stopping is not triggered).')
    parser.add_argument('--stop_criteria', type=int, default=6, help='training stop if epochs_since_improvement == stop_criteria')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--print_freq', type=int, default=100, help='print training/validation stats every __ batches.')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading; right now, only 0 works with h5pys in windows.')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning.')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='learning rate for decoder.')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of.')
    parser.add_argument('--alpha_c', type=float, default=1., help='regularization parameter for doubly stochastic attention, as in the paper.')
    parser.add_argument('--fine_tune_encoder', type=bool, default=True, help='whether fine-tune encoder or not')
    parser.add_argument('--fine_tune_embedding', type=bool, default=False, help='whether fine-tune word embeddings or not')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint, None if none.')
    parser.add_argument('--embedding_path', default=None, help='path to pre-trained word Embedding.')

    args = parser.parse_args()

    for encoder_layers, decoder_layers in [(3,3)]: #,,(0,6),(2,2),
        args.encoder_layers = encoder_layers
        args.decoder_layers = decoder_layers
        # args.encoder_mode = encoder_mode

        # load checkpoint, these parameters can't be modified
        final_args = {"emb_dim": args.emb_dim,
                     "attention_dim": args.attention_dim,
                     "decoder_dim": args.decoder_dim,
                     "n_heads": args.n_heads,
                     "dropout": args.dropout,
                     "decoder_mode": args.decoder_mode,
                     "attention_method": args.attention_method,
                     "encoder_layers": args.encoder_layers,
                     "decoder_layers": args.decoder_layers}

        start_epoch = 0
        best_bleu4 = 0.  # BLEU-4 score right now
        epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
        cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
        # print(device)

        # Read word map
        word_map_file = os.path.join(args.data_folder, 'WORDMAP_' + args.data_name + '.json')
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)

        # Initialize / load checkpoint
        if args.checkpoint is None:

            # Encoder
            encoder = CNN_Encoder(NetType=args.encoder_mode, attention_method=args.attention_method)

            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr) if args.fine_tune_encoder else None
            encoder_lr_scheduler = StepLR(encoder_optimizer,step_size=600,gamma=0.9)
            # set the encoder_dim
            encoder_dim = 512 if args.encoder_mode == 'vgg16' else 512 if args.encoder_mode == 'vgg19' \
                else 2048  # FIXME: encoder_dim depends on the model

            # different Decoder
            if args.decoder_mode == "lstm":
                decoder = DecoderLSTM(embed_dim=args.emb_dim,
                                      decoder_dim=args.decoder_dim,
                                      vocab_size=len(word_map),
                                      encoder_dim=encoder_dim,
                                      dropout=args.dropout)
            elif args.decoder_mode == "lstm_attention":
                decoder = DecoderWithAttention(attention_dim=args.attention_dim,
                                               embed_dim=args.emb_dim,
                                               decoder_dim=args.decoder_dim,
                                               vocab_size=len(word_map),
                                               encoder_dim=encoder_dim,
                                               dropout=args.dropout)
            elif args.decoder_mode == "transformer":
                decoder = Transformer(vocab_size=len(word_map),
                                      embed_dim=args.emb_dim,
                                      encoder_layers=args.encoder_layers,
                                      decoder_layers=args.decoder_layers,
                                      dropout=args.dropout,
                                      attention_method=args.attention_method,
                                      n_heads=args.n_heads)

            decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                 lr=args.decoder_lr)
            decoder_lr_scheduler = StepLR(decoder_optimizer,step_size=600,gamma=0.9)

            # load pre-trained word embedding
            if args.embedding_path is not None:
                all_word_embeds = {}
                for i, line in enumerate(codecs.open(args.embedding_path, 'r', 'utf-8')):
                    s = line.strip().split()
                    all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

                # change emb_dim
                args.emb_dim = list(all_word_embeds.values())[-1].size
                word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_map), args.emb_dim))
                for w in word_map:
                    if w in all_word_embeds:
                        word_embeds[word_map[w]] = all_word_embeds[w]
                    elif w.lower() in all_word_embeds:
                        word_embeds[word_map[w]] = all_word_embeds[w.lower()]
                    else:
                        # <pad> <start> <end> <unk>
                        embedding_i = torch.ones(1, args.emb_dim)
                        torch.nn.init.xavier_uniform_(embedding_i)
                        word_embeds[word_map[w]] = embedding_i

                word_embeds = torch.FloatTensor(word_embeds).to(device)
                decoder.load_pretrained_embeddings(word_embeds)
                decoder.fine_tune_embeddings(args.fine_tune_embedding)
                print('Loaded {} pre-trained word embeddings.'.format(len(word_embeds)))

        else:
            checkpoint = torch.load(args.checkpoint, map_location=str(device))
            start_epoch = checkpoint['epoch'] + 1
            epochs_since_improvement = checkpoint['epochs_since_improvement']
            best_bleu4 = checkpoint['metrics']["Bleu_4"]
            encoder = checkpoint['encoder']
            encoder_optimizer = checkpoint['encoder_optimizer']
            decoder = checkpoint['decoder']
            decoder_optimizer = checkpoint['decoder_optimizer']
            decoder.fine_tune_embeddings(args.fine_tune_embedding)
            # load final_args from checkpoint
            final_args = checkpoint['final_args']
            for key in final_args.keys():
                args.__setattr__(key, final_args[key])
            if args.fine_tune_encoder is True and encoder_optimizer is None:
                print("Encoder_Optimizer is None, Creating new Encoder_Optimizer!")
                encoder.fine_tune(args.fine_tune_encoder)
                encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                     lr=args.encoder_lr)

        # Move to GPU, if available
        decoder = decoder.to(device)
        encoder = encoder.to(device)
        print("Encoder_mode:{}   Decoder_mode:{}".format(args.encoder_mode,args.decoder_mode))
        print("encoder_layers {} decoder_layers {} n_heads {} dropout {} attention_method {} encoder_lr {} "
              "decoder_lr {} alpha_c {}".format(args.encoder_layers, args.decoder_layers, args.n_heads, args.dropout,
                                                args.attention_method, args.encoder_lr, args.decoder_lr, args.alpha_c))
        # print(encoder)
        # print(decoder)

        # Loss function
        criterion = nn.CrossEntropyLoss().to(device)

        # Custom dataloaders
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.399, 0.410, 0.371], std=[0.151, 0.138, 0.134])
        # normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        # pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        # If your data elements are a custom type, or your collate_fn returns a batch that is a custom type.
        train_loader = torch.utils.data.DataLoader(
            CaptionDataset(args.data_folder, args.data_name, 'TRAIN', transform=transforms.Compose([normalize])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            CaptionDataset(args.data_folder, args.data_name, 'TEST', transform=transforms.Compose([normalize])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

        # Epochs
        for epoch in range(start_epoch, args.epochs):

            # Decay learning rate if there is no improvement for 5 consecutive epochs, and terminate training after 25
            # 8 20
            if epochs_since_improvement == args.stop_criteria:
                print("the model has not improved in the last {} epochs".format(args.stop_criteria))
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 3 == 0:
                adjust_learning_rate(decoder_optimizer, 0.8)
                if args.fine_tune_encoder and encoder_optimizer is not None:
                    print(encoder_optimizer)
                    # adjust_learning_rate(encoder_optimizer, 0.8)

            # One epoch's training
            train(args,
                  train_loader=train_loader,
                  # val_loader=val_loader,
                  encoder=encoder,
                  decoder=decoder,
                  criterion=criterion,
                  encoder_optimizer=encoder_optimizer,
                  encoder_lr_scheduler=encoder_lr_scheduler,
                  decoder_optimizer=decoder_optimizer,
                  decoder_lr_scheduler=decoder_lr_scheduler,
                  epoch=epoch)


            # One epoch's validation
            metrics = validate(args,
                               val_loader=val_loader,
                               encoder=encoder,
                               decoder=decoder,
                               criterion=criterion)

            recent_bleu4 = metrics["Bleu_4"]

            # Check if there was an improvement
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            checkpoint_name = args.encoder_mode + '_' + args.decoder_mode + '_' + str(args.encoder_layers) + '_' + str(args.decoder_layers) + '_Res+MLAT' #_tengxun_aggregation
            save_checkpoint(checkpoint_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                            decoder_optimizer, metrics, is_best, final_args)
