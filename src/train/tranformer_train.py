''' This is the training file for Transformer'''
import time
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch import optim
from torch import cuda
from torch.autograd.variable import Variable
from train.model import Seq2Seq, docEmbedding
from train.preprocessing import data_iter
from train.settings import BATCH_SIZE, EMBEDDING_SIZE, GET_LOSS, GRAD_CLIP, PRETRAIN, SAVE_MODEL
from train.train import addpaddings, get_batch, sequenceloss
from train.util import gettime, load_model
from transformer import Transformer, Encoder, Decoder


'''
----- Lets use their data preperation? dataprepare.py ----
'''


# Fix arguments in constructor calls
# TODO: Rewrite to fit transformers 
def main(train_set, langs, embedding_size=EMBEDDING_SIZE,
          learning_rate=0.7, epochs=200, iter_num=None, pretrain=PRETRAIN, save_model=SAVE_MODEL):

    start = time.time()

    # TODO: Get Vocab Size, Prepare data
    emb = docEmbedding(langs['rt'].n_words, langs['re'].n_words,
                       langs['rm'].n_words, embedding_size)
    emb.init_weights()
    encoder = Transformer()# Need arguments
    decoder = Decoder()
    train_func = transform_train

    cuda_available = cuda.is_available

    if (cuda_available):
        emb.cuda()
        encoder.cuda()
        decoder.cuda()
    
    loss_optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                                lr=learning_rate)

    # Load pre-train model
    use_model = None
    if pretrain is not None and iter_num is not None:
        use_model = ['./models/' + pretrain + '_' + s + '_' + str(iter_num)
                     for s in ['encoder', 'decoder', 'optim']]

    if use_model is not None:
        encoder = load_model(encoder, use_model[0])
        decoder = load_model(decoder, use_model[1])
        loss_optimizer.load_state_dict(torch.load(use_model[2]))
        print("Load Pretrain Model {}".format(use_model))
    else:
        print("Not use Pretrain Model")

    criterion = nn.NLLLoss()
    
    model = Seq2Seq(encoder, decoder, train_func, criterion, embedding_size, langs)
    total_loss, iteration = 0, 0
    for epoch in epochs:
        print("Epoch: ", epoch)

        train_iter = data_iter(train_set, batch_size=BATCH_SIZE)
        for data_num in train_iter:
            iteration += 1
            data, idx_data = get_batch(data_num)
            rt, re, rm, summary = idx_data

            # Debugging: check the input triplets
            # show_triplets(data[0][0])

            # Add paddings
            rt = addpaddings(rt)
            re = addpaddings(re)
            rm = addpaddings(rm)
        
            summary = addpaddings(summary)

            rt = Variable(torch.LongTensor(rt), requires_grad=False)
            re = Variable(torch.LongTensor(re), requires_grad=False)
            rm = Variable(torch.LongTensor(rm), requires_grad=False)

            if cuda_available:
                rt, re, rm, summary = rt.cuda(), re.cuda(), rm.cuda(), summary.cuda()

            # Zero the gradient
            loss_optimizer.zero_grad()
            model.train()

            # calculate loss of "a batch of input sequence"
            loss = sequenceloss(rt, re, rm, summary, model)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm(list(model.encoder.parameters()) +
                                          list(model.decoder.parameters()),
                                          GRAD_CLIP)
            loss_optimizer.step()

            # Get the average loss on the sentences
            target_length = summary.size()[1]
            total_loss += loss.item() / target_length

            # Print the information and save model
            if iteration % GET_LOSS == 0:
                print("Time {}, iter {}, Seq_len:{}, avg loss = {:.4f}".format(
                    gettime(start), iteration, target_length, total_loss / GET_LOSS))
                total_loss = 0

        if epoch % save_model == 0:
            torch.save(encoder.state_dict(),
                       "models/{}_encoder_{}".format(output_file, iteration))
            torch.save(decoder.state_dict(),
                       "models/{}_decoder_{}".format(output_file, iteration))
            torch.save(loss_optimizer.state_dict(),
                       "models/{}_optim_{}".format(output_file, iteration))
            print("Save the model at iter {}".format(iteration))

    return model.encoder, model.decoder

            
def transform_train(model, device, optimizer, train_loader, lr, epoch, log_interval):
    model.train()
    losses = []
    hidden = None
    for batch_idx, (data, label) in enumerate(tqdm.tqdm(train_loader)):
        data, label = data.to(device), label.to(device)
        # Separates the hidden state across batches.
        # Otherwise the backward would try to go all the way to the beginning every time.
        if hidden is not None:
            hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data)
        pred = output.max(-1)[1]
        loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)


def transform_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        hidden = None
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, hidden)
            test_loss += model.loss(output, label, reduction='mean').item()
            pred = output.max(-1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            # Comment this out to avoid printing test results
            if batch_idx % 10 == 0:
                print('Input\t%s\nGT\t%s\npred\t%s\n\n' % (
                    test_loader.dataset.vocab.array_to_words(data[0]),
                    test_loader.dataset.vocab.array_to_words(label[0]),
                    test_loader.dataset.vocab.array_to_words(pred[0])))

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / \
        (len(test_loader.dataset) * test_loader.dataset.sequence_length)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) *
        test_loader.dataset.sequence_length,
        100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)))
    return test_loss, test_accuracy


