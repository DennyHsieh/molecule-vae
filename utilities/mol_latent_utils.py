import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AllChem as Chem


def chk_target_prop_df(prop_df, homo_max, lumo_min):
    return prop_df[(prop_df['homo'] <= homo_max) & (prop_df['lumo'] >= lumo_min)]


def match_target_prop_df(real_vals, pred_vals, property_column, homo_max, lumo_min):
    prop_pred_df = pd.DataFrame(pred_vals, columns=property_column)
    prop_real_df = pd.DataFrame(real_vals, columns=property_column)

    prop_pred_match_df = chk_target_prop_df(prop_pred_df, homo_max, lumo_min)
    prop_real_match_df = chk_target_prop_df(prop_real_df, homo_max, lumo_min)

    print('Predict data matches target properties: {:.4f}({}/{})'.format(
        len(prop_pred_match_df) / len(prop_pred_df), len(prop_pred_match_df), len(prop_pred_df)))
    print('Real data matches target properties: {:.4f}({}/{})'.format(
        len(prop_real_match_df) / len(prop_real_df), len(prop_real_match_df), len(prop_real_df)))
    return prop_pred_match_df, prop_real_match_df


def same(list1, list2):
    list_dif = [i for i in list1 if i in list2]
    return list_dif


def difference(list1, list2):
    list_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
    return list_dif


def perturb_z(z, noise_norm, constant_norm=False):
    if noise_norm > 0.0:
        noise_vec = np.random.normal(0, 1, size=z.shape)
        noise_vec = noise_vec / np.linalg.norm(noise_vec)
        if constant_norm:
            return z + (noise_norm * noise_vec)
        else:
            noise_amp = np.random.uniform(0, noise_norm, size=(z.shape[0], 1))
            return z + (noise_amp * noise_vec)
    else:
        return z


# Shows linear inteprolation in image space vs latent space
def interpolation_z(latentStart, latentEnd, nbSteps=5):
    print("Generating interpolations...")
    vectors = []
    # Linear interpolation
    alphaValues = np.linspace(0, 1, nbSteps)
    for alpha in alphaValues:
        # Latent space interpolation
        vector = latentStart * (1 - alpha) + latentEnd * alpha
        vectors.append(vector.unsqueeze(0))

    # Decode latent space vectors
    vectors = torch.cat(vectors)
    # reconstructions = decoder(latent=vectors, max_len=100)
    return vectors


# === Molecule convert ===
def smiles_to_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        pass
    return None


def matched_ring(s):
    return s.count('1') % 2 == 0 and s.count('2') % 2 == 0


def balanced_parentheses(input_string):
    s = []
    balanced = True
    index = 0
    while index < len(input_string) and balanced:
        token = input_string[index]
        if token == "(":
            s.append(token)
        elif token == ")":
            if len(s) == 0:
                balanced = False
            else:
                s.pop()
        index += 1
    return balanced and len(s) == 0


def fast_verify(s):
    return matched_ring(s) and balanced_parentheses(s)


def smiles_distance_z(z_rep, z0):
    # x = self.smiles_to_hot(smiles)
    # z_rep = self.encode(x)
    return np.linalg.norm(z0 - z_rep, axis=1)


def prep_mol_df(smiles, prop_np_vals=None, prop_cols=None):
    df = pd.DataFrame({'smiles': smiles})
    if prop_cols is not None:
        prop_df = pd.DataFrame(prop_np_vals, columns=prop_cols)
        prop_df = pd.concat([df, prop_df], axis=1)
        prop_df = prop_df.groupby('smiles').mean()
    sort_df = pd.DataFrame(df[['smiles']].groupby(by='smiles').size().rename('count').reset_index())
    df = df.merge(sort_df, on='smiles')
    if prop_cols is not None:
        df = df.merge(prop_df, on='smiles')
    df.drop_duplicates(subset='smiles', inplace=True)
    # df = df[df['smiles'].apply(fast_verify)]
    df['mol'] = df['smiles'].apply(smiles_to_mol)
    if len(df) > 0:
        df = df[pd.notnull(df['mol'])]
    if len(df) > 0:
        # df['distance'] = smiles_distance_z(df['smiles'], z)
        # df['frequency'] = df['count'] / float(sum(df['count']))
        # df = df[['smiles', 'distance', 'count', 'frequency', 'mol']]
        # df.sort_values(by='distance', inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df
