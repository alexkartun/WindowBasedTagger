from abc import abstractmethod
import matplotlib.pyplot as plt
from torch import optim, nn
import torch.nn.functional as F
import utils as ut
import numpy as np
import torch.utils.data

STUDENT = {'name': 'Alex Kartun_Ofir Sharon',
           'ID': '324429216_204717664'}

# tagger and nn model globals
default_batch_size = 1
batch_size = 1024
embedding_dim = 50
context_size = 5


class Net(nn.Module):
    """
    nn net class implementing mlp with one hidden layer and a tanh activation function and passing output
    through a softmax transformation, the setting weights of the embeddings vectors is depends on the include
    embedding boolean
    """
    def __init__(self):
        super(Net, self).__init__()
        if not inc_embedding:
            # just creating embedding vectors without preprocessed weights
            self.embeddings = nn.Embedding(len(ut.W2I), embedding_dim)
        else:
            # generating embedding vectors from the file and copy the weight to the embedding vectors
            embedding_vecs = ut.generate_embedding_vecs()
            self.embeddings = nn.Embedding(embedding_vecs.shape[0], embedding_dim)
            self.embeddings.weight.data.copy_(torch.from_numpy(embedding_vecs))
        # hidden layers
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, len(ut.T2I))

    def forward(self, inputs):
        # converting the shape's of the embedding matrix to be fit to the hidden layer
        embeds = self.embeddings(inputs).view(-1, context_size * embedding_dim)
        out = torch.tanh(self.linear1(embeds))
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)


class SpecialNet(Net):
    """
    nn net class extending from net, same implementation as the father but,
    thr logic of this net is summing prefixes and suffixes vectors with the embeddings vectors before passing
    to activation function, seems like we adding more power
    """
    def __init__(self):
        super(SpecialNet, self).__init__()
        # creating of prefix and suffix embeddings vectors which will be added to the embeddings vectors
        self.prefix_embeddings = nn.Embedding(len(ut.P2I), embedding_dim)
        self.suffix_embeddings = nn.Embedding(len(ut.S2I), embedding_dim)

    def forward(self, inputs):
        # extracting prefixes and suffixes vectors from the inputs vectors
        inputs_copy = inputs.data.numpy().reshape(-1)
        prefixes = np.asanyarray([ut.P2I[ut.I2W[word_index][:ut.sub_word_units_size]] for word_index in inputs_copy])
        suffixes = np.asanyarray([ut.S2I[ut.I2W[word_index][-ut.sub_word_units_size:]] for word_index in inputs_copy])
        prefixes = torch.from_numpy(prefixes.reshape(inputs.data.shape)).type(torch.LongTensor)
        suffixes = torch.from_numpy(suffixes.reshape(inputs.data.shape)).type(torch.LongTensor)
        # summing up all vectors
        embeds = (self.embeddings(inputs) + self.prefix_embeddings(prefixes) +
                  self.suffix_embeddings(suffixes)).view(-1, context_size * embedding_dim)
        out = torch.tanh(self.linear1(embeds))
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)


class Tagger(object):
    """
    abstract tagging class with abstract generate_accuracy_and_loss_on_dev function
    """
    def __init__(self, nn_model):
        self.train_loader = tr_loader
        self.dev_loader = dv_loader
        self.test_loader = tst_loader
        self.nn_model = nn_model
        self.optimizer = optim.Adam(self.nn_model.parameters(), lr=lr)
        self.dev_accuracy_per_epoch = {}
        self.dev_loss_per_epoch = {}

    def run(self):
        """
        starting to run the tagger
        :return:
        """
        self.train()
        self.plot_dev_data()
        self.output_pred(self.predict())

    def train(self):
        """
        training the model and each epoch save the loss and accuracy on dev data loader in the maps
        :return:
        """
        print('starting to train the model...')
        self.nn_model.train()
        for epoch in range(epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                net_out = self.nn_model(data)
                loss = F.nll_loss(net_out, target)
                loss.backward()
                self.optimizer.step()
            dev_accuracy, dev_loss = self.generate_accuracy_and_loss_on_dev()
            self.dev_accuracy_per_epoch[epoch] = dev_accuracy
            self.dev_loss_per_epoch[epoch] = dev_loss

    @abstractmethod
    def generate_accuracy_and_loss_on_dev(self):
        """
        abstract function, children should implement
        :return:
        """
        pass

    def predict(self):
        """
        predicting the labels of test loader by the trained model
        :return: list of predicted tags
        """
        print('starting to predict the labels by the model...')
        pred_lst = []
        self.nn_model.eval()
        for data in self.test_loader:
            net_out = self.nn_model(data)
            pred = net_out.data.max(1)[1]
            pred_lst.append(ut.I2T[pred.item()])
        return pred_lst

    def plot_dev_data(self):
        """
        creating 2 plots: 1st plot of dev accuracy to number of epochs and 2nd fot dev loss to number of epochs
        :return:
        """
        plt.plot(self.dev_accuracy_per_epoch.keys(), self.dev_accuracy_per_epoch.values(), 'r-')
        plt.savefig(accuracy_filename)
        plt.clf()
        plt.plot(self.dev_loss_per_epoch.keys(), self.dev_loss_per_epoch.values(), 'r-')
        plt.savefig(loss_filename)

    @staticmethod
    def output_pred(pred_list):
        """
        writing the predicted tags to prediction file, the format is: word<space>pred_tag
        :param pred_list: predicted tags
        :return:
        """
        print('writing the tagged output to the file...')
        output = []
        blank_lines = 0
        with open(test_filename, 'r') as file:
            for line_ind, line in enumerate(file.readlines()):
                if not line.strip():
                    blank_lines += 1
                    output.append('')
                    continue
                word = line.strip()
                output.append('{} {}'.format(word, pred_list[line_ind - blank_lines]))
        with open(pred_filename, 'w') as file:
            file.write('\n'.join(output))


class PosTagger(Tagger):
    """
    pos tagging class extending from tagger class
    this class implementing his generate_accuracy_and_loss_on_dev function
    """
    def generate_accuracy_and_loss_on_dev(self):
        """
        generating accuracy and loss on dev data loader
        :return: dev accuracy and dev loss
        """
        dev_loss = 0
        correct = 0
        for data, target in self.dev_loader:
            net_out = self.nn_model(data)
            dev_loss += F.nll_loss(net_out, target).item()
            pred = net_out.data.max(1)[1]
            correct += pred.eq(target.data).sum()
        dev_loss /= len(self.dev_loader.dataset)
        dev_accuracy = (100. * correct.item()) / len(self.dev_loader.dataset)
        print('Dev set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(dev_loss, correct,
                                                                                len(self.dev_loader.dataset),
                                                                                dev_accuracy))
        return dev_accuracy, dev_loss


class NerTagger(Tagger):
    """
    ner tagging class extending from tagger class
    this class implementing his generate_accuracy_and_loss_on_dev function
    """
    def generate_accuracy_and_loss_on_dev(self):
        """
        generating accuracy and loss on dev data loader by skipping correct outputs of 'O'
        :return: dev accuracy and dev loss
        """
        dev_loss = 0
        correct = 0
        total = 0
        for data, target in self.dev_loader:
            net_out = self.nn_model(data)
            dev_loss += F.nll_loss(net_out, target).item()
            pred = net_out.data.max(1)[1]
            if ut.I2T[pred.item()] != 'O' or ut.I2T[target.item()] != 'O':
                correct += pred.eq(target.data).sum()
                total += 1
        dev_loss /= len(self.dev_loader.dataset)
        dev_accuracy = (100. * correct.item()) / total
        print('Dev set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(dev_loss,
                                                                                correct, total, dev_accuracy))
        return dev_accuracy, dev_loss


def generate_input_loader(features_vecs, targets_vecs, batch_si):
    """
    creating data loader of train/dev features vectors and targets vectors
    :param features_vecs: features vectors
    :param targets_vecs: targets vectors
    :param batch_si: batch size
    :return: return generated data loader
    """
    data = torch.from_numpy(np.asarray(features_vecs)).type(torch.LongTensor)
    targets = torch.from_numpy(np.asarray(targets_vecs)).type(torch.LongTensor)
    dset = torch.utils.data.TensorDataset(data, targets)
    return torch.utils.data.DataLoader(dset, batch_si, shuffle=True)


def generate_validation_loader(features_vecs):
    """
    creating torch tensor of test features vectors
    :param features_vecs: features vecs
    :return: torch tensor of features vecs
    """
    return torch.from_numpy(np.asarray(features_vecs)).type(torch.LongTensor)


def preprocess_data():
    """
    preprocessing the data and creating train, dev and test loaders. the creation of vocabulary
    depends on include embedding boolean
    :return: train, dev and test loader
    """
    print('generating train, dev and test data...')
    if not inc_embedding:
        ut.generate_util_sets(train_filename)
    else:
        ut.generate_words_set()
        ut.generate_tags_set(train_filename)
    ut.generate_util_maps()
    features_vecs, targets_vecs = ut.generate_input_data(train_filename)
    train_loader = generate_input_loader(features_vecs, targets_vecs, batch_size)
    features_vecs, targets_vecs = ut.generate_input_data(dev_filename)
    dev_loader = generate_input_loader(features_vecs, targets_vecs, default_batch_size)
    features_vecs = ut.generate_validation_data(test_filename)
    test_loader = generate_validation_loader(features_vecs)
    return train_loader, dev_loader, test_loader


def generate_tagger():
    """
    generating tagger and model, the creation of the model depends on include sub word units boolean
    :return: nn model and tagger
    """
    print('creating the model and tagger...')
    if inc_sub_word_units:
        # for the third task this net will be created that extending from NET
        ut.generete_extra_util_maps()
        model = SpecialNet()
    else:
        # for the 1st and second task this net will be created
        model = Net()
    if tagger_type == 'pos':
        tagg = PosTagger(model)
    else:
        tagg = NerTagger(model)
    return tagg


if __name__ == '__main__':
    # extracting data from config and generating global variables
    with open('config', 'r') as f:
        task = f.readline().strip()
        tagger_type = f.readline().strip()
        lr = float(f.readline().strip())
        epochs = int(f.readline().strip())
        hidden_dim = int(f.readline().strip())
        inc_embedding = f.readline().strip() == '1'
        inc_sub_word_units = f.readline().strip() == '1'
    train_filename = '{}/{}'.format(tagger_type, 'train')
    dev_filename = '{}/{}'.format(tagger_type, 'dev')
    test_filename = '{}/{}'.format(tagger_type, 'test')
    pred_filename = '{}/{}{}.{}'.format(tagger_type, 'test', task, tagger_type)
    accuracy_filename = '{}/{}{}.{}'.format(tagger_type, 'acc_dev', task, 'png')
    loss_filename = '{}/{}{}.{}'.format(tagger_type, 'loss_dev', task, 'png')
    # generating loaders, model, tagger
    tr_loader, dv_loader, tst_loader = preprocess_data()
    tagger = generate_tagger()
    # run the tagger
    tagger.run()
