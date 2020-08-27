
import tqdm
import os
import sys
sys.path.append(".")

import torch
import time
from rdkit import Chem
import pandas as pd

from base_classes import schedulers
from data.molecule_iterator import SmileBucketIterator
from vae import vae_models
from vae.vae_trainer import VAETrainer, VAEArgParser


if __name__ == '__main__':
    args = VAEArgParser().parse_args()
    print(args)
    run_predictor = True
    property_column = ['gap', 'IE', 'EA']
    # property_column = ['gap', 'homo', 'lumo']
    # property_column = ['logP', 'qed', 'SAS']

    # smi_file = 'data/zinc.smi'
    # vocab_file = 'data/vacab'

    # rdkit load SMILES and structures
    dset_path = 'data'
    esol_dset_path = os.path.join(dset_path, 'ESOL')
    esol_dset = os.path.join(esol_dset_path, 'delaney-processed.csv')
    zinc_dset = os.path.join(dset_path, '250k_rndm_zinc_drugs_clean_3.csv')

    # Polymer genome Dset
    kha_dset = os.path.join(dset_path, 'khazana_db_DFT.csv')
    kha_df = pd.read_csv(kha_dset, index_col=None)
    kha_df = kha_df.rename(columns={"SMILES String": "smiles", "Band Gap, PBE": "gap", "Ionization Energy": "IE",
                                    "Electron Affinity": "EA"})
    # kha_df = kha_df.rename(columns={"SMILES String": "smiles", "Band Gap, HSE06": "gap", "Ionization Energy": "IE",
    #                                 "Electron Affinity": "EA"})
    kha_df = kha_df.dropna(subset=['smiles', 'gap'])
    kha_df['smiles'] = kha_df['smiles'].str.strip()
    kha_df['smiles_len'] = kha_df['smiles'].apply(len)
    kha_df['gap'] = kha_df['gap'].map(lambda x: x.strip(' eV')).astype(float)
    kha_df['IE'] = kha_df['IE'].astype(str).map(lambda x: x.strip(' eV')).astype(float)
    kha_df['EA'] = kha_df['EA'].astype(str).map(lambda x: x.strip(' eV')).astype(float)

    smi_file = kha_df
    vocab_file = 'data/vacab_kha'

    # # QM9 Dataset
    # qm9_dset_path = os.path.join(dset_path, 'qm9')
    # qm9_dset = os.path.join(qm9_dset_path, 'gdb9.sdf.csv')
    # qm9_dset_suppl = os.path.join(qm9_dset_path, 'gdb9.sdf')
    # qm9_suppl = Chem.SDMolSupplier(qm9_dset_suppl)
    # # qm9_mols = [x for x in qm9_suppl]
    #
    # qm9_smiles = dict()
    # qm9_null_val = 0
    # for idx, x in enumerate(qm9_suppl, start=1):
    #     gdb_idx = "gdb_" + str(idx)
    #     try:
    #         qm9_smiles[gdb_idx] = Chem.MolToSmiles(x)
    #     except:
    #         qm9_smiles[gdb_idx] = None
    #         qm9_null_val += 1
    #         continue
    #
    # print('# of DB is {} . After MolToSmiles, NULL val is {}'.format(len(qm9_suppl), qm9_null_val))
    # qm9_smiles_df = pd.DataFrame(list(qm9_smiles.items()), columns=['mol_id', 'smiles'])
    # qm9_df = pd.read_csv(qm9_dset)
    # qm9_df = pd.merge(qm9_df, qm9_smiles_df, on='mol_id', how='inner')
    # qm9_df = qm9_df.dropna()
    # qm9_df['smiles'] = qm9_df['smiles'].str.strip()
    #
    # smi_file = qm9_df
    # vocab_file = 'data/vacab_qm9'
    #
    # smi_iterator = SmileBucketIterator(smi_file, vocab_file, args.batch_size, property_column=['gap'])

    # # ZINC Dataset
    # zinc_df = pd.read_csv(zinc_dset, index_col=None)
    # zinc_df['smiles'] = zinc_df['smiles'].str.strip()
    #
    # smi_file = zinc_df
    # vocab_file = 'data/vacab_zinc'

    smi_iterator = SmileBucketIterator(smi_file, vocab_file, args.batch_size, property_column=property_column)
    
    if args.test_mode:
        smi_iterator.train_smi.examples = smi_iterator.train_smi.examples[:1000]

    train_bucket_iter = smi_iterator.train_bucket_iter()
    test_bucket_iter = smi_iterator.test_bucket_iter()
    vocab_size = smi_iterator.vocab_size
    padding_idx = smi_iterator.padding_idx
    sos_idx = smi_iterator.sos_idx
    eos_idx = smi_iterator.eos_idx
    unk_idx = smi_iterator.unk_idx
    vocab = smi_iterator.get_vocab()
    print('vocab_size:', vocab_size)
    print('padding_idx sos_idx eos_idx unk_idx:', padding_idx, sos_idx, eos_idx, unk_idx)
    print(vocab.itos, vocab.stoi)

    # define Vae model
    vae = vae_models.Vae(vocab, vocab_size, args.embedding_size, args.dropout,
                         padding_idx, sos_idx, unk_idx,
                         args.max_len, args.n_layers, args.layer_size, smi_iterator.prop_size, pred_depth=3,
                         bidirectional=args.enc_bidir, latent_size=args.latent_size, partialsmiles=args.partialsmiles,
                         run_predictor=run_predictor)
    
    enc_optimizer = torch.optim.Adam(vae.encoder_params, lr=3e-4)
    dec_optimizer = torch.optim.Adam(vae.decoder_params, lr=1e-4)
    if run_predictor:
        pred_optimizer = torch.optim.Adam(vae.predictor_params, lr=1e-4)
    else: pred_optimizer = None

    #scheduler = schedulers.Scheduler(enc_optimizer, dec_optimizer, 0.5, 1e-8)
    scheduler = schedulers.StepScheduler(enc_optimizer, dec_optimizer, epoch_anchors=[200, 250, 275])
    vae = vae.cuda()
    trainer = VAETrainer(args, vocab, vae, enc_optimizer, dec_optimizer, scheduler, train_bucket_iter, test_bucket_iter,
                         pred_optimizer, prop_weight=10)

    if args.generate_samples:
        if not args.restore:
            raise ValueError('argument --restore with trained vae path required to generate samples!')
        trainer.load_raw(args.restore)
        # random sampling
        samples = []
        for _ in tqdm.tqdm(range(10)):
            samples.append(trainer.sample_prior(1000).cpu())
        samples = torch.cat(samples, 0)
        torch.save(samples, 'prior_samples.pkl')
    else:
        # validate
        if args.restore:
            trainer.load_raw(args.restore)
        else:
            trainer.train()
        print('recon_acc:', trainer.validate(epsilon_std=1e-6))

