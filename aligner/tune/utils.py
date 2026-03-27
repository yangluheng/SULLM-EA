import numpy as np
import torch
import random, os


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def print_trainable_params(first_model, model):
    trainable_params = 0
    all_param = 0

    for _, param in first_model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW
    elif optim == 'adamax':
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer

def output_decode(eval_output, eval_label, tokenizer):
    eval_decode_output = []
    eval_decode_label = []
    assert len(eval_output) == len(eval_label)
    for i in range(len(eval_output)):
        batch_output = eval_output[i]
        label_output = eval_label[i]
        eval_decode_output.extend(tokenizer.batch_decode(batch_output, skip_special_tokens=True))
        eval_decode_label.extend(tokenizer.batch_decode(label_output, skip_special_tokens=True))
    assert len(eval_decode_label) == len(eval_decode_output)

    return eval_decode_output, eval_decode_label


def get_edge_num(x):
    return len(x['edge_index'][0])


def read_raw_data(file_dir, l=[1, 2], reverse=False):
    print("loading raw data...")
    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups

    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids

    ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in l])
    ills = read_file([file_dir + "/ill_ent_ids"])
    triples = read_file([file_dir + "/triples_" + str(i) for i in l])
    rel_size = max([t[1] for t in triples]) + 1
    reverse_triples = []
    r_hs, r_ts = {}, {}
    if len(triples[0]) <= 3:
        for h, r, t in triples:
            if r not in r_hs:
                r_hs[r] = set()
            if r not in r_ts:
                r_ts[r] = set()
            r_hs[r].add(h)
            r_ts[r].add(t)
            if reverse:
                reverse_r = r + rel_size
                reverse_triples.append((t, reverse_r, h))
                if reverse_r not in r_hs:
                    r_hs[reverse_r] = set()
                if reverse_r not in r_ts:
                    r_ts[reverse_r] = set()
                r_hs[reverse_r].add(t)
                r_ts[reverse_r].add(h)
    else:
        for h, r, t, t1, t2 in triples:
            if r not in r_hs:
                r_hs[r] = set()
            if r not in r_ts:
                r_ts[r] = set()
            r_hs[r].add(h)
            r_ts[r].add(t)
            if reverse:
                reverse_r = r + rel_size
                reverse_triples.append((t, reverse_r, h))
                if reverse_r not in r_hs:
                    r_hs[reverse_r] = set()
                if reverse_r not in r_ts:
                    r_ts[reverse_r] = set()
                r_hs[reverse_r].add(t)
                r_ts[reverse_r].add(h)
    if reverse:
        triples = triples + reverse_triples
    assert len(r_hs) == len(r_ts)
    return ent2id_dict, ills, triples, r_hs, r_ts, ids
