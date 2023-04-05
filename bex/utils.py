from tqdm import tqdm
import os
import requests
import numpy as np
import torch

# bound perturbations to lr-ball
def bound_z(z_perturbed, latents, char_embed, font_embed):

    """Bound the perturbed latent variables to maintain a valid counterfactual explanation.

    Args:
        z_perturbed (``torch.Tensor``): The perturbed latent variables.
        latents (``torch.Tensor``): The original latent variables.
        char_embed (``torch.Tensor``): The character embeddings.
        font_embed (``torch.Tensor``): The font embeddings.

    Returns:
        ``torch.Tensor``: The bounded perturbed latent variables.
    """

    b, ne , c = z_perturbed.size()
    latents = latents[:, None, :].repeat(1, ne, 1).view(-1, c)
    z_perturbed = z_perturbed.view(-1 ,c)
    delta = z_perturbed - latents

    r_cont = 1.
    norm = torch.linalg.norm(delta[:, -5:], 1, -1) + 1e-6
    r = torch.ones_like(norm) * r_cont
    delta[:, -5:] = torch.minimum(norm, r)[:, None] * delta[:, -5:] / norm[:, None]
    delta = torch.minimum(norm, r)[:, None] * delta / norm[:, None]

    # character
    r_char = torch.cdist(char_embed, char_embed, p=1).max()
    norm = torch.linalg.norm(delta[:, :3], 1, -1) + 1e-6
    r = torch.ones_like(norm) * r_char
    delta[:, :3] = torch.minimum(norm, r)[:, None] * delta[:, :3] / norm[:, None]

    # font
    r_font = torch.cdist(font_embed, font_embed, p=1).max()
    norm = torch.linalg.norm(delta[:, 3:6], 1, -1) + 1e-6
    r = torch.ones_like(norm) * r_font
    delta[:, 3:6] = torch.minimum(norm, r)[:, None] * delta[:, 3:6] / norm[:, None]

    z_perturbed = latents + delta

    return z_perturbed.view(b, ne, c)


def get_successful_cf(z_perturbed, z, perturbed_logits, logits, char_embed, font_embed):

    """Get successful counterfactual explanations and related metrics.

    Args:
        z_perturbed (``torch.Tensor``): The perturbed latent variables.
        z (``torch.Tensor``): The original latent variables.
        perturbed_logits (``torch.Tensor``): The logits of the perturbed examples.
        logits (``torch.Tensor``): The logits of the original examples.
        char_embed (``torch.Tensor``): The character embeddings.
        font_embed (``torch.Tensor``): The font embeddings.

    Returns:
        ``tuple``: Successful counterfactual explanations, SCE (Successful Counterfactual Explanations),
               non-causal flip, causal flip, and trivial explanations.
    """

    b, ne, c = z_perturbed.size()
    z = z[:, None, :].repeat(1, ne, 1).view(b, ne, c)
    logits = logits[:, None, :].repeat(1, ne, 1).view(b * ne, -1)
    z_perturbed = transform_embedding(z_perturbed, char_embed, font_embed)

    z = transform_embedding(z, char_embed, font_embed)
    perturbed_preds = z_perturbed.view(-1, z_perturbed.size(-1))[:, :48].argmax(1)
    preds = z.view(-1, z.size(-1))[:, :48].argmax(1)

    oracle = torch.zeros_like(preds)
    perturbed_oracle = torch.zeros_like(preds)
    oracle[preds % 2 == 1] = 1
    perturbed_oracle[perturbed_preds % 2 == 1] = 1

    classifier = logits.argmax(1).view(b, ne)
    perturbed_classifier = perturbed_logits.argmax(1).view(b, ne)
    oracle = oracle.view(b, ne)
    perturbed_oracle = perturbed_oracle.view(b, ne)

    E_cc = perturbed_classifier != classifier
    E_causal_change = ~E_cc & (perturbed_oracle != oracle)
    E_agree = (classifier == oracle) & (perturbed_classifier == perturbed_oracle)
    successful_cf = (E_cc | E_causal_change) & ~E_agree

    ncfs = successful_cf.sum() if successful_cf.sum() !=0 else 1
    if successful_cf.sum() != 0:
        sce = successful_cf.float().mean().item()
        non_causal_flip = ((E_cc & ~E_agree).sum() / ncfs).item()
        causal_flip = ((E_causal_change & ~E_agree).sum() / ncfs).item()
        trivial = ((E_cc & E_agree).float().mean().item())
    else:
        sce = 0
        non_causal_flip = 0
        causal_flip = 0
        trivial = 0

    return successful_cf, sce, non_causal_flip, causal_flip, trivial


def normalize_embeddings(embedding, mean, std):

    embedding = (embedding - mean) / std

    return embedding


def transform_embedding(z, char_embed, font_embed, one_hot=False):

    """Transform embedding for a given set of variables to a interpretable space.

    Parameters:
        z (``torch.Tensor``): The latent variables.
        char_embed (``torch.Tensor``): The character embeddings.
        font_embed (``torch.Tensor``): The font embeddings.
        one_hot (``bool``, optional): Whether to one-hot the output. Defaults to False.

    Returns:
        ``torch.Tensor``: The cosine embedded latent variables.
    """

    b, ne, c = z.size()
    z = z.view(-1, c)
    char_embed = char_embed.to(z.device)
    font_embed = font_embed.to(z.device)

    # first 3 are the embedding of char class
    # the next 3 font embedding
    z_char = z[:, None, :3]
    z_font = z[:, None, 3:6]
    char = torch.softmax((torch.linalg.norm(char_embed[None, ...] - z_char, 2, dim=-1) * -3), dim=-1)
    font = torch.softmax((torch.linalg.norm(font_embed[None, ...] - z_font, 2, dim=-1) * -3), dim=-1)

    if one_hot:
        char_max = char.argmax(-1)
        font_max = font.argmax(-1)
        char = torch.zeros(b * ne, 48).to(z.device)
        char[torch.arange(b * ne), char_max] = 1
        font = torch.zeros(b * ne, 48).to(z.device)
        font[torch.arange(b* ne), font_max] = 1

    z = torch.cat((char, font, z[:, -5:]), 1)

    return z.view(b, ne, -1)


def compute_metrics(z, z_perturbed, successful_cf, char_embed, font_embed):

    if successful_cf.sum() == 0:
        return 0, None

    b, ne, c = z_perturbed.size()
    z = z[:, None, :].repeat(1, ne, 1).view(b, ne, -1).detach()
    z = transform_embedding(z, char_embed, font_embed, one_hot=True)
    z_perturbed = transform_embedding(z_perturbed, char_embed, font_embed)

    z_perturbed = z_perturbed.view_as(z).detach()

    success = []
    idxs = []
    for i, samples in enumerate(successful_cf):

        if z[i][samples].numel() == 0:
            continue

        ortho_set = torch.tensor([]).to(z_perturbed.device)
        norm = torch.linalg.norm(z[i][samples] - z_perturbed[i][samples], ord=1, dim=-1)
        norm_sort = torch.argsort(norm)
        z_perturbed_sorted = z_perturbed[i][norm_sort]
        z_sorted = z[i][norm_sort]

        idx = []
        for j, (exp, latent) in enumerate(zip(z_perturbed_sorted, z_sorted)):

            exp[-5:] = latent[-5:] - exp[-5:]
            exp[:-5] = exp[:-5] * (1 - latent[:-5])
            exp = exp[None]

            if ortho_set.numel() == 0:
                idx.append(j)
                ortho_set = torch.cat((ortho_set, exp), 0)

            else:
                cos_discrete = torch.cosine_similarity(exp[:, :-5], ortho_set[:, :-5])
                cos_cont = torch.cosine_similarity(exp[:, -5:], ortho_set[:, -5:])
                cos_sim = cos_discrete + cos_cont
                # tau=0.15
                if torch.all(cos_sim.abs() < 0.15) or torch.any(cos_sim < -1 + 0.15):

                    idx.append(j)
                    ortho_set = torch.cat((ortho_set, exp), 0)

        success.append(ortho_set.size(0))
        idxs.append((i, samples.nonzero().view(-1)[norm_sort[idx]]))


    return np.mean(success), idxs



def get_data_path_or_download(dataset, data_root, download):

    """Finds a dataset locally and downloads if needed.

    Args:
        dataset (str): dataset name. For instance 'camouflage_n=100000_2020-Oct-19.h5py'.
            See https://github.com/ElementAI/synbols-resources/tree/master/datasets/generated for the complete list. (please ignore .a[a-z] extensions)
        data_root (str): path where the dataset will be or is stored. If empty string, it defaults to $TMPDIR

    Raises:
        ValueError: dataset name does not exist in local path nor in remote

    Returns:
        str: dataset final path
    """

    MIRRORS = ['https://zenodo.org/record/6616598/files/%s?download=1', ]

    if data_root == "":
        data_root = os.environ.get("TMPDIR", "/tmp")
    full_path = os.path.join(data_root, dataset)

    if os.path.isfile(full_path) and not download:
        print(f"{full_path} found")
        return full_path
    else:
        print(f"Downloading {full_path}...")

    for i, mirror in enumerate(MIRRORS):
        try:
            download_url(mirror % dataset, full_path)
            return full_path
        except Exception as e:
            if i + 1 < len(MIRRORS):
                print("%s failed, trying %s..." %(mirror, MIRRORS[i+1]))
            else:
                raise e


def download_url(url, path):
    r = requests.head(url)
    if r.status_code == 302:
        raise RuntimeError("Server returned 302 status. Try again later or contact us.")

    if not r.ok:
        raise ValueError(f"File {url}, not found in remote.")

    # Check if the file already exists
    if os.path.isfile(path):
        # Get the size of the existing file
        existing_file_size = os.path.getsize(path)
    else:
        existing_file_size = 0

    # If the existing file size is equal to the total file size, skip the download process and return
    total_size_in_bytes = int(r.headers.get('content-length', 0))
    if existing_file_size == total_size_in_bytes:
        print(f"File {path} already downloaded.")
        return

    headers = {"Range": f"bytes={existing_file_size}-"} if existing_file_size else {}
    response = requests.get(url, stream=True, headers=headers)
    block_size = 1024 #1 Kilobyte
    progress_bar = tqdm(total=total_size_in_bytes, initial=existing_file_size, unit='iB', unit_scale=True)
    with open(path, 'ab' if existing_file_size else 'wb') as file:
        if existing_file_size:
            print(f"Resume downloading from: {url}")
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            file.close()
            os.remove(path)

            raise RuntimeError(f"ERROR, something went wrong downloading {url}")

# taken from haven-ai

def hash_dict(exp_dict):


    """Create a hash for an experiment.

    Parameters
    ----------
    exp_dict : dict
        An experiment, which is a single set of hyper-parameters

    Returns
    -------
    hash_id: str
        A unique id defining the experiment
    """
    dict2hash = ""
    if not isinstance(exp_dict, dict):
        raise ValueError("exp_dict is not a dict")

    for k in sorted(exp_dict.keys()):
        if "." in k:
            raise ValueError(". has special purpose")
        elif isinstance(exp_dict[k], dict):
            v = hash_dict(exp_dict[k])
        elif isinstance(exp_dict[k], tuple):
            raise ValueError(f"{exp_dict[k]} tuples can't be hashed yet, consider converting tuples to lists")
        elif isinstance(exp_dict[k], list) and len(exp_dict[k]) and isinstance(exp_dict[k][0], dict):
            v_str = ""
            for e in exp_dict[k]:
                if isinstance(e, dict):
                    v_str += hash_dict(e)
                else:
                    raise ValueError("all have to be dicts")
            v = v_str
        else:
            v = exp_dict[k]

        dict2hash += str(k) + "/" + str(v)
    hash_id = hashlib.md5(dict2hash.encode()).hexdigest()

    return hash_id
