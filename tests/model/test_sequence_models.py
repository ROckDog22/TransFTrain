import sys
sys.path.append('./python')
import numpy as np
import pytest
import torch
import unittest

import TransFTrain as train
import TransFTrain.nn as nn

from simple_training import *
from models import LanguageModel


np.random.seed(3)

BATCH_SIZES = [1, 15]
INPUT_SIZES = [1, 11]
HIDDEN_SIZES = [1, 12]
BIAS = [True, False]
INIT_HIDDEN = [True, False]
NONLINEARITIES = ['tanh', 'relu']
SEQ_LENGTHS = [1, 13]
NUM_LAYERS = [1, 2]
OUTPUT_SIZES = [1, 1000]
EMBEDDING_SIZES = [1, 34]
SEQ_MODEL = ['rnn', 'lstm']

class TestSequenceModel(unittest.TestCase):
    def test_rnn_cell_cpu(self):
        device = train.cpu()
        for batch_size in BATCH_SIZES:
            for input_size in INPUT_SIZES:
                for hidden_size in HIDDEN_SIZES:
                    for bias in BIAS:
                        for init_hidden in INIT_HIDDEN:
                            for nonlinearity in NONLINEARITIES:
                                x = np.random.randn(batch_size, input_size).astype(np.float32)
                                h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

                                model_ = torch.nn.RNNCell(input_size, hidden_size, nonlinearity=nonlinearity, bias=bias)
                                if init_hidden:
                                    h_ = model_(torch.tensor(x), torch.tensor(h0))
                                else:
                                    h_ = model_(torch.tensor(x), None)

                                model = nn.RNNCell(input_size, hidden_size, device=device, bias=bias, nonlinearity=nonlinearity)
                                model.W_ih = train.Tensor(model_.weight_ih.detach().numpy().transpose(), device=device)
                                model.W_hh = train.Tensor(model_.weight_hh.detach().numpy().transpose(), device=device)
                                if bias:
                                    model.bias_ih = train.Tensor(model_.bias_ih.detach().numpy(), device=device)
                                    model.bias_hh = train.Tensor(model_.bias_hh.detach().numpy(), device=device)
                                if init_hidden:
                                    h = model(train.Tensor(x, device=device), train.Tensor(h0, device=device))
                                else:
                                    h = model(train.Tensor(x, device=device), None)
                                assert h.device == device
                                np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
                                h.sum().backward()
                                h_.sum().backward()
                                np.testing.assert_allclose(model_.weight_ih.grad.detach().numpy().transpose(), model.W_ih.grad.numpy(), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_rnn_cell_cuda(self):
        device = train.cuda()
        for batch_size in BATCH_SIZES:
            for input_size in INPUT_SIZES:
                for hidden_size in HIDDEN_SIZES:
                    for bias in BIAS:
                        for init_hidden in INIT_HIDDEN:
                            for nonlinearity in NONLINEARITIES:
                                x = np.random.randn(batch_size, input_size).astype(np.float32)
                                h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

                                model_ = torch.nn.RNNCell(input_size, hidden_size, nonlinearity=nonlinearity, bias=bias)
                                if init_hidden:
                                    h_ = model_(torch.tensor(x), torch.tensor(h0))
                                else:
                                    h_ = model_(torch.tensor(x), None)

                                model = nn.RNNCell(input_size, hidden_size, device=device, bias=bias, nonlinearity=nonlinearity)
                                model.W_ih = train.Tensor(model_.weight_ih.detach().numpy().transpose(), device=device)
                                model.W_hh = train.Tensor(model_.weight_hh.detach().numpy().transpose(), device=device)
                                if bias:
                                    model.bias_ih = train.Tensor(model_.bias_ih.detach().numpy(), device=device)
                                    model.bias_hh = train.Tensor(model_.bias_hh.detach().numpy(), device=device)
                                if init_hidden:
                                    h = model(train.Tensor(x, device=device), train.Tensor(h0, device=device))
                                else:
                                    h = model(train.Tensor(x, device=device), None)
                                assert h.device == device
                                np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
                                h.sum().backward()
                                h_.sum().backward()
                                np.testing.assert_allclose(model_.weight_ih.grad.detach().numpy().transpose(), model.W_ih.grad.numpy(), atol=1e-5, rtol=1e-5)


    def test_lstm_cell_cpu(self):
        device = train.cpu()
        for batch_size in BATCH_SIZES:
            for input_size in INPUT_SIZES:
                for hidden_size in HIDDEN_SIZES:
                    for bias in BIAS:
                        for init_hidden in INIT_HIDDEN:
                            x = np.random.randn(batch_size, input_size).astype(np.float32)
                            h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
                            c0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

                            model_ = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)
                            if init_hidden:
                                h_, c_ = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
                            else:
                                h_, c_ = model_(torch.tensor(x), None)

                            model = nn.LSTMCell(input_size, hidden_size, device=device, bias=bias)

                            model.W_ih = train.Tensor(model_.weight_ih.detach().numpy().transpose(), device=device)
                            model.W_hh = train.Tensor(model_.weight_hh.detach().numpy().transpose(), device=device)
                            if bias:
                                model.bias_ih = train.Tensor(model_.bias_ih.detach().numpy(), device=device)
                                model.bias_hh = train.Tensor(model_.bias_hh.detach().numpy(), device=device)

                            if init_hidden:
                                h, c = model(train.Tensor(x, device=device), (train.Tensor(h0, device=device), train.Tensor(c0, device=device)))
                            else:
                                h, c = model(train.Tensor(x, device=device), None)
                            np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
                            np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)

                            h.sum().backward()
                            h_.sum().backward()
                            np.testing.assert_allclose(model_.weight_ih.grad.detach().numpy().transpose(), model.W_ih.grad.numpy(), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_lstm_cell_cuda(self):
        device = train.cuda()
        for batch_size in BATCH_SIZES:
            for input_size in INPUT_SIZES:
                for hidden_size in HIDDEN_SIZES:
                    for bias in BIAS:
                        for init_hidden in INIT_HIDDEN:
                            x = np.random.randn(batch_size, input_size).astype(np.float32)
                            h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
                            c0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

                            model_ = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)
                            if init_hidden:
                                h_, c_ = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
                            else:
                                h_, c_ = model_(torch.tensor(x), None)

                            model = nn.LSTMCell(input_size, hidden_size, device=device, bias=bias)

                            model.W_ih = train.Tensor(model_.weight_ih.detach().numpy().transpose(), device=device)
                            model.W_hh = train.Tensor(model_.weight_hh.detach().numpy().transpose(), device=device)
                            if bias:
                                model.bias_ih = train.Tensor(model_.bias_ih.detach().numpy(), device=device)
                                model.bias_hh = train.Tensor(model_.bias_hh.detach().numpy(), device=device)

                            if init_hidden:
                                h, c = model(train.Tensor(x, device=device), (train.Tensor(h0, device=device), train.Tensor(c0, device=device)))
                            else:
                                h, c = model(train.Tensor(x, device=device), None)
                            np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
                            np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)

                            h.sum().backward()
                            h_.sum().backward()
                            np.testing.assert_allclose(model_.weight_ih.grad.detach().numpy().transpose(), model.W_ih.grad.numpy(), atol=1e-5, rtol=1e-5)

    def test_rnn_cpu(self):
        device = train.cpu()
        for seq_length in SEQ_LENGTHS:
            for num_layers in NUM_LAYERS:
                for batch_size in BATCH_SIZES:
                    for input_size in INPUT_SIZES:
                        for hidden_size in HIDDEN_SIZES:
                            for bias in BIAS:
                                for init_hidden in INIT_HIDDEN:
                                    for nonlinearity in NONLINEARITIES:
                                        x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
                                        h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

                                        model_ = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers, bias=bias, nonlinearity=nonlinearity)
                                        if init_hidden:
                                            output_, h_ = model_(torch.tensor(x), torch.tensor(h0))
                                        else:
                                            output_, h_ = model_(torch.tensor(x), None)

                                        model = nn.RNN(input_size, hidden_size, num_layers, bias, device=device, nonlinearity=nonlinearity)
                                        for k in range(num_layers):
                                            model.rnn_cells[k].W_ih = train.Tensor(getattr(model_, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
                                            model.rnn_cells[k].W_hh = train.Tensor(getattr(model_, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
                                            if bias:
                                                model.rnn_cells[k].bias_ih = train.Tensor(getattr(model_, f'bias_ih_l{k}').detach().numpy(), device=device)
                                                model.rnn_cells[k].bias_hh = train.Tensor(getattr(model_, f'bias_hh_l{k}').detach().numpy(), device=device)
                                        if init_hidden:
                                            output, h = model(train.Tensor(x, device=device), train.Tensor(h0, device=device))
                                        else:
                                            output, h = model(train.Tensor(x, device=device), None)

                                        np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
                                        np.testing.assert_allclose(output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5)

                                        output.sum().backward()
                                        output_.sum().backward()
                                        np.testing.assert_allclose(model.rnn_cells[0].W_ih.grad.detach().numpy(), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)


    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_rnn_cuda(self):
        device = train.cuda()
        for seq_length in SEQ_LENGTHS:
            for num_layers in NUM_LAYERS:
                for batch_size in BATCH_SIZES:
                    for input_size in INPUT_SIZES:
                        for hidden_size in HIDDEN_SIZES:
                            for bias in BIAS:
                                for init_hidden in INIT_HIDDEN:
                                    for nonlinearity in NONLINEARITIES:
                                        x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
                                        h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

                                        model_ = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers, bias=bias, nonlinearity=nonlinearity)
                                        if init_hidden:
                                            output_, h_ = model_(torch.tensor(x), torch.tensor(h0))
                                        else:
                                            output_, h_ = model_(torch.tensor(x), None)

                                        model = nn.RNN(input_size, hidden_size, num_layers, bias, device=device, nonlinearity=nonlinearity)
                                        for k in range(num_layers):
                                            model.rnn_cells[k].W_ih = train.Tensor(getattr(model_, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
                                            model.rnn_cells[k].W_hh = train.Tensor(getattr(model_, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
                                            if bias:
                                                model.rnn_cells[k].bias_ih = train.Tensor(getattr(model_, f'bias_ih_l{k}').detach().numpy(), device=device)
                                                model.rnn_cells[k].bias_hh = train.Tensor(getattr(model_, f'bias_hh_l{k}').detach().numpy(), device=device)
                                        if init_hidden:
                                            output, h = model(train.Tensor(x, device=device), train.Tensor(h0, device=device))
                                        else:
                                            output, h = model(train.Tensor(x, device=device), None)

                                        np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
                                        np.testing.assert_allclose(output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5)

                                        output.sum().backward()
                                        output_.sum().backward()
                                        np.testing.assert_allclose(model.rnn_cells[0].W_ih.grad.detach().numpy(), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)


    def test_lstm_cpu(self):
        device = train.cpu()
        for seq_length in SEQ_LENGTHS:
            for num_layers in NUM_LAYERS:
                for batch_size in BATCH_SIZES:
                    for input_size in INPUT_SIZES:
                        for hidden_size in HIDDEN_SIZES:
                            for bias in BIAS:
                                for init_hidden in INIT_HIDDEN:
                                    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
                                    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
                                    c0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

                                    model_ = torch.nn.LSTM(input_size, hidden_size, bias=bias, num_layers=num_layers)
                                    if init_hidden:
                                        output_, (h_, c_) = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
                                    else:
                                        output_, (h_, c_) = model_(torch.tensor(x), None)

                                    model = nn.LSTM(input_size, hidden_size, num_layers, bias, device=device)
                                    for k in range(num_layers):
                                        model.lstm_cells[k].W_ih = train.Tensor(getattr(model_, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
                                        model.lstm_cells[k].W_hh = train.Tensor(getattr(model_, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
                                        if bias:
                                            model.lstm_cells[k].bias_ih = train.Tensor(getattr(model_, f'bias_ih_l{k}').detach().numpy(), device=device)
                                            model.lstm_cells[k].bias_hh = train.Tensor(getattr(model_, f'bias_hh_l{k}').detach().numpy(), device=device)
                                    if init_hidden:
                                        output, (h, c) = model(train.Tensor(x, device=device), (train.Tensor(h0, device=device), train.Tensor(c0, device=device)))
                                    else:
                                        output, (h, c) = model(train.Tensor(x, device=device), None)

                                    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
                                    np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)
                                    np.testing.assert_allclose(output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5)

                                    output.sum().backward()
                                    output_.sum().backward()
                                    np.testing.assert_allclose(model.lstm_cells[0].W_ih.grad.detach().numpy(), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_lstm_cuda(self):
        device = train.cuda()
        for seq_length in SEQ_LENGTHS:
            for num_layers in NUM_LAYERS:
                for batch_size in BATCH_SIZES:
                    for input_size in INPUT_SIZES:
                        for hidden_size in HIDDEN_SIZES:
                            for bias in BIAS:
                                for init_hidden in INIT_HIDDEN:
                                    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
                                    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
                                    c0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

                                    model_ = torch.nn.LSTM(input_size, hidden_size, bias=bias, num_layers=num_layers)
                                    if init_hidden:
                                        output_, (h_, c_) = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
                                    else:
                                        output_, (h_, c_) = model_(torch.tensor(x), None)

                                    model = nn.LSTM(input_size, hidden_size, num_layers, bias, device=device)
                                    for k in range(num_layers):
                                        model.lstm_cells[k].W_ih = train.Tensor(getattr(model_, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
                                        model.lstm_cells[k].W_hh = train.Tensor(getattr(model_, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
                                        if bias:
                                            model.lstm_cells[k].bias_ih = train.Tensor(getattr(model_, f'bias_ih_l{k}').detach().numpy(), device=device)
                                            model.lstm_cells[k].bias_hh = train.Tensor(getattr(model_, f'bias_hh_l{k}').detach().numpy(), device=device)
                                    if init_hidden:
                                        output, (h, c) = model(train.Tensor(x, device=device), (train.Tensor(h0, device=device), train.Tensor(c0, device=device)))
                                    else:
                                        output, (h, c) = model(train.Tensor(x, device=device), None)

                                    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
                                    np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)
                                    np.testing.assert_allclose(output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5)

                                    output.sum().backward()
                                    output_.sum().backward()
                                    np.testing.assert_allclose(model.lstm_cells[0].W_ih.grad.detach().numpy(), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)

    def test_language_model_implementation_cpu(self):
        device = train.cpu()
        for seq_length in SEQ_LENGTHS:
            for num_layers in NUM_LAYERS:
                for batch_size in BATCH_SIZES:
                    for embedding_size in EMBEDDING_SIZES:
                        for hidden_size in HIDDEN_SIZES:
                            for init_hidden in INIT_HIDDEN:
                                for output_size in OUTPUT_SIZES:
                                    for seq_model in SEQ_MODEL:
                                        #TODO add test for just nn.embedding?
                                        x = np.random.randint(0, output_size, (seq_length, batch_size)).astype(np.float32)
                                        h0 = train.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)
                                        c0 = train.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)

                                        model = LanguageModel(embedding_size, output_size, hidden_size, num_layers, seq_model, device=device)
                                        if init_hidden:
                                            if seq_model == 'lstm':
                                                h = (h0, c0)
                                            elif seq_model == 'rnn':
                                                h = h0
                                            output, h_ = model(train.Tensor(x, device=device), h)
                                        else:
                                            output, h_ = model(train.Tensor(x, device=device), None)

                                        if seq_model == 'lstm':
                                            assert isinstance(h_, tuple)
                                            h0_, c0_ = h_
                                            assert c0_.shape == (num_layers, batch_size, hidden_size)
                                        elif seq_model == 'rnn':
                                            h0_ = h_
                                        assert h0_.shape == (num_layers, batch_size, hidden_size)
                                        assert output.shape == (batch_size * seq_length, output_size)
                                        #TODO actually test values
                                        output.backward()
                                        for p in model.parameters():
                                            assert p.grad is not None

    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_language_model_implementation_cuda(self):
        device = train.cuda()
        for seq_length in SEQ_LENGTHS:
            for num_layers in NUM_LAYERS:
                for batch_size in BATCH_SIZES:
                    for embedding_size in EMBEDDING_SIZES:
                        for hidden_size in HIDDEN_SIZES:
                            for init_hidden in INIT_HIDDEN:
                                for output_size in OUTPUT_SIZES:
                                    for seq_model in SEQ_MODEL:
                                        #TODO add test for just nn.embedding?
                                        x = np.random.randint(0, output_size, (seq_length, batch_size)).astype(np.float32)
                                        h0 = train.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)
                                        c0 = train.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)

                                        model = LanguageModel(embedding_size, output_size, hidden_size, num_layers, seq_model, device=device)
                                        if init_hidden:
                                            if seq_model == 'lstm':
                                                h = (h0, c0)
                                            elif seq_model == 'rnn':
                                                h = h0
                                            output, h_ = model(train.Tensor(x, device=device), h)
                                        else:
                                            output, h_ = model(train.Tensor(x, device=device), None)

                                        if seq_model == 'lstm':
                                            assert isinstance(h_, tuple)
                                            h0_, c0_ = h_
                                            assert c0_.shape == (num_layers, batch_size, hidden_size)
                                        elif seq_model == 'rnn':
                                            h0_ = h_
                                        assert h0_.shape == (num_layers, batch_size, hidden_size)
                                        assert output.shape == (batch_size * seq_length, output_size)
                                        #TODO actually test values
                                        output.backward()
                                        for p in model.parameters():
                                            assert p.grad is not None


    def test_language_model_training_cpu(self):
        device = train.cpu()
        np.random.seed(0)
        corpus = train.data.Corpus("data/ptb", max_lines=20)
        seq_len = 10
        num_examples = 100
        batch_size = 16
        seq_model = 'rnn'
        num_layers = 2
        hidden_size = 10
        n_epochs=2
        train_data = train.data.batchify(corpus.train, batch_size=batch_size, device=device, dtype="float32")
        model = LanguageModel(30, len(corpus.dictionary), hidden_size=hidden_size, num_layers=num_layers, seq_model=seq_model, device=device)
        train_acc, train_loss = train_ptb(model, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device)
        test_acc, test_loss = evaluate_ptb(model, train_data, seq_len=seq_len, device=device)
        if str(device) == "cpu()":
            np.testing.assert_allclose(5.711512, train_loss, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(5.388685, test_loss, atol=1e-5, rtol=1e-5)
        elif str(device) == "cuda()":
            np.testing.assert_allclose(5.711512, train_loss, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(5.388685, test_loss, atol=1e-5, rtol=1e-5)


    @unittest.skipIf(not train.cuda().enabled(), "NO GPU")
    def test_language_model_training_cuda(self):
        device = train.cuda()
        np.random.seed(0)
        corpus = train.data.Corpus("data/ptb", max_lines=20)
        seq_len = 10
        num_examples = 100
        batch_size = 16
        seq_model = 'rnn'
        num_layers = 2
        hidden_size = 10
        n_epochs=2
        train_data = train.data.batchify(corpus.train, batch_size=batch_size, device=device, dtype="float32")
        model = LanguageModel(30, len(corpus.dictionary), hidden_size=hidden_size, num_layers=num_layers, seq_model=seq_model, device=device)
        train_acc, train_loss = train_ptb(model, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device)
        test_acc, test_loss = evaluate_ptb(model, train_data, seq_len=seq_len, device=device)
        if str(device) == "cpu()":
            np.testing.assert_allclose(5.711512, train_loss, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(5.388685, test_loss, atol=1e-5, rtol=1e-5)
        elif str(device) == "cuda()":
            np.testing.assert_allclose(5.711512, train_loss, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(5.388685, test_loss, atol=1e-5, rtol=1e-5)



if __name__ == '__main__':
    unittest.main()