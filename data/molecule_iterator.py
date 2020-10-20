import os
import pickle
import re
import torch
import torchtext
from torchtext.data import Example, Field, Dataset
from torchtext.data import BucketIterator

pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


def smi_tokenizer(smi, regex=re.compile(pattern)):
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens), 'smi:' + smi + '--tokens:' + ''.join(tokens)
    return tokens


class SmileBucketIterator(object):
    def __init__(self, data_file, vocab_file, batch_size=256, property_column=None):
        self.batch_size = batch_size

        smi_field = Field(sequential=True, init_token='<sos>', eos_token=' ',
                          pad_token=' ', include_lengths=True, batch_first=True, tokenize=smi_tokenizer)
        property_field = Field(sequential=False, use_vocab=False, dtype=torch.float)
        # load smile data
        # with open(data_file, 'r') as f:
        #     mol_strs = f.read().strip().split('\n')
        #     mol_strs = [mol.replace(' ', '') for mol in mol_strs]
        # mol_strs = [smi_field.preprocess(mol) for mol in mol_strs]
        mol_strs = []
        smi_examples = []
        fields = [('smile', smi_field), ('property', property_field)]
        for index, row in data_file.iterrows():
            mol_str = smi_field.preprocess(row['smiles'])
            # prop_str = property_field.preprocess(row[property_column].tolist())
            if property_column is not None:
                ex = Example.fromlist([mol_str, row[property_column].tolist()], fields)
            else:
                ex = Example.fromlist([mol_str, [0]], fields)
            mol_strs.append(mol_str)
            smi_examples.append(ex)

        # load or build vocab
        if os.path.isfile(vocab_file):
            print('load vocab from:', vocab_file)
            smi_field.vocab = pickle.load(open(vocab_file, 'rb'))
        else:
            print('build and save vocab file:', vocab_file)
            smi_field.build_vocab(mol_strs)
            pickle.dump(smi_field.vocab, open(vocab_file, 'wb'), protocol=2)

        self.dset_num = len(mol_strs)
        # self.dset_test_num = int(self.dset_num * 0.2)
        self.vocab = smi_field.vocab
        self.vocab_size = len(smi_field.vocab.itos)
        if property_column is not None:
            self.prop_size = len(property_column)
        else:
            self.prop_size = 0
        self.padding_idx = smi_field.vocab.stoi[smi_field.pad_token]
        self.sos_idx = smi_field.vocab.stoi[smi_field.init_token]
        self.eos_idx = smi_field.vocab.stoi[smi_field.eos_token]
        self.unk_idx = smi_field.vocab.stoi[smi_field.unk_token]

        self.dataset_smi = Dataset(smi_examples, fields=fields)
        self.train_smi, self.test_smi = self.dataset_smi.split(0.8)
        self.dset_test_num = len(self.test_smi)
        # self.train_smi = Dataset(smi_examples[:-self.dset_test_num], fields=fields)
        # self.test_smi = Dataset(smi_examples[-self.dset_test_num:], fields=fields)

    def dataset_bucket_iter(self, batch_size=None):
        bsize = self.batch_size if batch_size is None else batch_size
        return BucketIterator(self.dataset_smi, batch_size=bsize, train=False, shuffle=False,
                              sort=False, sort_within_batch=False, repeat=False)

    def train_bucket_iter(self, batch_size=None):
        bsize = self.batch_size if batch_size is None else batch_size
        return BucketIterator(self.train_smi, batch_size=bsize, train=True,
                              sort_within_batch=True, repeat=False, sort_key=lambda x: len(x.smile))

    def test_bucket_iter(self, batch_size=None):
        bsize = self.batch_size if batch_size is None else batch_size
        return BucketIterator(self.test_smi, batch_size=bsize, train=False,
                              sort_within_batch=True, repeat=False, sort_key=lambda x: len(x.smile))

    def get_vocab(self):
        return self.vocab
