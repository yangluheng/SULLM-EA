import json
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from utils import get_edge_num, read_raw_data

from conversation import conv_templates, SeparatorStyle


IGNORE_TOKEN_ID = -100


def get_instructions(args, data_path):
    df = pd.read_json(data_path)
    df['edge_num'] = df.apply(get_edge_num, axis=1)
    df['gpt'] = df['output']
    df = df.reset_index(drop=True)
    if not args.inference:
        df = df.sample(frac=1, ignore_index=True)
    elif args.data_num != 'all':
        df = df.sample(n=int(args.data_num), ignore_index=True)
    else:
        df = df.sample(frac=1, ignore_index=True)
    return df


def preprocess(instruction, tokenizer, max_length, mode='train'):
    """
    预处理函数，将 instruction 构建成 LLM 可用的输入张量。

    参数：
    ----------
    instruction : dict
        单条指令数据，包含 "prompt" 和 "gpt" 等键。
    tokenizer : PreTrainedTokenizer
        用于将文本转换成 token id。
    max_length : int
        输入序列的最大长度。
    mode : str
        'train' 或 'eval' 模式。训练模式需要构建 mask，推理模式不需要。

    返回：
    ----------
    dict 包含以下键：
        input_ids : torch.Tensor
            token id 输入。
        target_ids : torch.Tensor
            mask 之后的训练目标，训练时只计算模型生成部分的 loss。
        attention_mask : torch.Tensor
            attention mask。
        text : str
            构建好的 prompt 文本。
    """
    # === 1选择 Vicuna 模板 ===
    conv = conv_templates["vicuna_v1_1"].copy()
    assert conv.sep_style == SeparatorStyle.TWO  # 检查模板分隔符风格
    roles = conv.roles  # 获取角色名称，例如 ["User", "Assistant"]

    # 不同模式下，padding 方向不同
    tokenizer.padding_side = 'right' if mode == 'train' else 'left'

    # === 2构建对话 prompt ===
    conversations = []
    conv.append_message(roles[0], instruction["prompt"])  # 用户输入
    if mode == 'train':
        conv.append_message(roles[1], instruction["gpt"])  # 模型回答
    else:
        conv.append_message(roles[1], None)  # 推理时回答为空
    conversations.append(conv.get_prompt())  # 获取完整对话文本
    # print(conversations)
    # === 3使用 tokenizer 编码输入 ===
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids


    # === 4 构建训练目标 target_ids ===
    if mode != 'train':
        # 推理模式下，target_ids 是模型回答的 token id
        end_token = "</s>"
        targets = tokenizer(
            [instruction["gpt"] + end_token],
            return_tensors="pt",
            padding="max_length",
            max_length=100,
            truncation=True,
        ).input_ids
    else:
        # 训练模式下，target_ids 需要 mask prompt 部分，只计算回答部分的 loss
        targets = input_ids.clone()
        sep = conv.sep + conv.roles[1] + ": "  # 角色分隔符，用于找到回答开始位置

        # 遍历每条对话
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 实际 token 长度
            turns = conversation.split(conv.sep2)  # 按对话轮次分隔
            cur_len = 1  # 跳过初始 token
            target[:cur_len] = IGNORE_TOKEN_ID  # mask 开始 token

            # 遍历每轮对话
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct. the first label is not _, but _label
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    instruction_len -= 1

                # Ignore the user instructions
                target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len

                if i != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            # mask 后续 pad 部分
            target[cur_len:] = IGNORE_TOKEN_ID

            # 检查长度匹配
            if cur_len < max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" #turn = {len(turns) - 1}. (ignored)"
                    )

    return dict(
        input_ids=input_ids,  # 模型输入 token id
        target_ids=targets,  # mask 后的目标 token id
        attention_mask=input_ids.ne(tokenizer.pad_token_id),  # attention mask
        text=conversations[0]  # prompt 文本
    )


class KG_InstructionDataset_EA(Dataset):
   
    def __init__(self, tokenizer, args, mode='test') -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.mode = mode

        if args.inference:
            path = f'{args.instruction_path}/{args.test_dataset}/kg_instruction_{args.test_dataset}_{self.mode}.json'

        else:
            path = f'{args.instruction_path}/{args.dataset}/kg_instruction_{args.dataset}_{self.mode}.json'

        self.instructions = get_instructions(args, path)

        self.kg = torch.load(f"./data/kg/kg_{args.dataset}_data.pt")

        self.name2data = self.kg

        # 使用 Xavier / Glorot 初始化
        def xavier_init(num_nodes, feat_dim, device=self.args.gpu):
            tensor = torch.empty(num_nodes, feat_dim, device=device)
            return nn.init.xavier_uniform_(tensor)

        def get_embeddings(file_dir, img_path, attr_emb_path_1, attr_emb_path_2, e, topR=1000, lang_list=[1, 2]):
            ent2id_dict, ills, triples, r_hs, r_ts, ids = (
                read_raw_data(file_dir, lang_list)
            )


            attr_1 = json.load(open(attr_emb_path_1, "r"))
            attr_2 = json.load(open(attr_emb_path_2, "r"))

            attr_dict = {**attr_1, **attr_2}  # 最干净最安全
            print("KG1 attr size:", len(attr_1))
            print("KG2 attr size:", len(attr_2))
            print("Merged size:", len(attr_dict))

            img_dict = pickle.load(open(img_path, "rb"))
            neighbor_img = defaultdict(list)
            neighbor_attr = defaultdict(list)

            for triple in triples:
                # 图像邻居
                head = triple[0]
                relation = triple[1]
                tail = triple[2]
                if tail in img_dict:
                    neighbor_img[head].append(tail)
                if head in img_dict:
                    neighbor_img[tail].append(head)

                # 属性邻居
                if str(tail) in attr_dict:
                    neighbor_attr[str(head)].append(str(tail))
                if str(head) in attr_dict:
                    neighbor_attr[str(tail)].append(str(head))

            imgs_np = np.array(list(img_dict.values()))
            entities_with_images = len(set(img_dict.keys()))
            print("entities_with_images: ", entities_with_images)
            img_mean = np.mean(imgs_np, axis=0)
            img_std = np.std(imgs_np, axis=0)
            global_img_emb = np.random.normal(img_mean, img_std, img_mean.shape[0])

            length_counter = Counter()

            for eid, emb in attr_dict.items():
                if isinstance(emb, (list, tuple, np.ndarray)):
                    length_counter[len(emb)] += 1
                else:
                    length_counter['scalar_or_other'] += 1

            print("Embedding length distribution in attr_dict:")
            for k, v in length_counter.items():
                print(f"Length {k}: {v} entities")

            for eid, emb in attr_dict.items():
                if not isinstance(emb, (list, tuple, np.ndarray)):
                    print("Invalid embedding:", eid, emb)

            if len(attr_dict) > 0:
                valid_attrs = [v for v in attr_dict.values() if isinstance(v, (list, np.ndarray))]
                if not valid_attrs:
                    print("WARNING: all attribute embeddings are invalid!")
                    global_attr_emb = None
                else:
                    attr_mean = np.mean(np.stack(valid_attrs), axis=0)
                    attr_std = np.std(np.stack(valid_attrs), axis=0)
                    global_attr_emb = np.random.normal(attr_mean, attr_std, attr_mean.shape[0])

                    for k, v in attr_dict.items():
                        if v is None or len(v) != len(attr_mean):
                            attr_dict[str(k)] = attr_mean

                    attrs_np = np.stack(list(attr_dict.values()))
                    entities_with_attrs = len(attr_dict)
                    print("entities_with_attrs: ", entities_with_attrs)
            else:
                print("WARNING: no attribute embeddings found!")
                global_attr_emb = None

            img_embd = []
            attr_embd = []

            follow_nei_img = 0
            follow_all_img = 0
            follow_nei_attr = 0
            follow_all_attr = 0

            for i in range(e):  # i = 0 ... num_entities-1
                if i in img_dict:
                    img_embd.append(img_dict[i])
                else:
                    if len(neighbor_img[i]) > 0:
                        follow_nei_img += 1
                        nei_imgs = np.array([img_dict[n] for n in neighbor_img[i]])
                        img_embd.append(np.mean(nei_imgs, axis=0))
                    else:
                        follow_all_img += 1
                        img_embd.append(global_img_emb)

                if str(i) in attr_dict:
                    attr_embd.append(np.array(attr_dict[str(i)]))
                else:
                    if len(neighbor_attr[str(i)]) > 0:
                        follow_nei_attr += 1
                        nei_attrs = np.array([attr_dict[str(n)] for n in neighbor_attr[str(i)]])
                        attr_embd.append(np.mean(nei_attrs, axis=0))
                    else:
                        follow_all_attr += 1
                        attr_embd.append(global_attr_emb)

            print(
                "%.2f%% entities have images," % (100 * len(img_dict) / e),
                " follow_nei_img:", follow_nei_img,
                " follow_all_img:", follow_all_img,
            )
            print(
                "%.2f%% entities have attrs," % (100 * len(attr_dict) / e),
                " follow_nei_attr:", follow_nei_attr,
                " follow_all_attr:", follow_all_attr,
            )

            img_embeddings = np.array(img_embd)
            attr_embeddings = np.array(attr_embd)

            return torch.tensor(img_embeddings, dtype=torch.float32, device=self.args.gpu), torch.tensor(attr_embeddings,
                                                                                                  dtype=torch.float32,
                                                                                                  device=self.args.gpu), None

        num_nodes = self.name2data[args.dataset]['total_entities']

        path = args.dataset_path + args.dataset + "/"

        img_path = path + args.dataset + "_id_img_feature_dict.pkl"
        attr_path1 = path + "attr_summary_emb_1.json"
        attr_path2 = path + "attr_summary_emb_2.json"


        if args.single_modal:

            print("Single-modal：transe实体嵌入")
            initial_embeddings = torch.load(f"./data/kg/{args.dataset}_initial_embeddings.pt", map_location=args.gpu)

            initial_embeddings = initial_embeddings.detach().clone()
            self.name2data[args.dataset].update({
                'x': initial_embeddings
            })


        args.gnn_input = self.name2data[args.dataset]['x'].shape[1]
        args.edge_dim = None
        print(f"[EA] entity feature dim: {args.gnn_input}")



    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        raw = self.instructions.iloc[idx]
        instruction = raw.copy()
        tokens = " ".join([f"[Entity {i}]" for i in range(1, 1 + self.args.num_token)])
        data_name = instruction['data']

        instruction['prompt'] = instruction['prompt'].replace("[Entity 1]", tokens)


        out_dict = preprocess(instruction, self.tokenizer, self.args.max_text_length, self.mode)

        # 构造图数据
        graph = Data()
        graph.edge_index = torch.LongTensor(instruction['edge_index'])
        node_list = torch.LongTensor(instruction['node_set'])


        graph.x = self.name2data[data_name]['x'][node_list].to(dtype=torch.bfloat16)
        if self.args.single_modal:
            graph.vis_feat = None
            graph.attr_feat = None
            graph.rel_feat = None

        graph.ea = True if instruction['task'] == 'ea' else False
        graph.edge_attr = None
        is_node = (out_dict['input_ids'] >= 32000)
        out_dict['is_node'] = is_node
        out_dict['graph'] = graph

        return out_dict

    def collate_fn(self, batch):
        """支持模态缺失的 batch 拼接"""
        batch_entry = {}
        input_ids, target_ids, attn_mask, is_node, graph = [], [], [], [], []


        for i, entry in enumerate(batch):
            input_ids.append(entry['input_ids'])
            target_ids.append(entry['target_ids'])
            attn_mask.append(entry['attention_mask'])
            is_node.append(entry['is_node'])
            graph.append(entry['graph'])

        batch_entry['input_ids'] = torch.cat(input_ids, dim=0)
        batch_entry['target_ids'] = torch.cat(target_ids, dim=0)
        batch_entry['attn_mask'] = torch.cat(attn_mask, dim=0)
        batch_entry['is_node'] = torch.cat(is_node, dim=0)
        batch_entry['graph'] = Batch.from_data_list(graph)

        return batch_entry

