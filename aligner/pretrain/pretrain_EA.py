import json
from collections import Counter, defaultdict

import swanlab
import torch
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
import pickle
import random
import os
import os.path as osp
import pandas as pd
import torch.nn.functional as F
import torch_geometric

from utils.utils import read_raw_data
from model.model import GraphSAGE
from dataloader import NodeNegativeLoader
from loss.contrastive_loss import ContrastiveLoss, GraceLoss


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(args, data, model, optimizer, criterion):
    train_loader = NodeNegativeLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        neg_ratio=args.num_negs,
        num_neighbors=fans_out,
        mask_feat_ratio_1=args.drop_feature_rate_1,
        mask_feat_ratio_2=args.drop_feature_rate_2,
        drop_edge_ratio_1=args.drop_edge_rate_1,
        drop_edge_ratio_2=args.drop_edge_rate_2,
    )
    model.train()
    total_loss = 0
    total_ins_loss = 0
    total_con_loss = 0
    total_multi_loss = 0

    pbar = tqdm(total=len(train_loader))
    for step, (ori_graph, view_1, view_2) in enumerate(train_loader):
        ori_graph, view_1, view_2 = ori_graph.to(device), view_1.to(device), view_2.to(device)
        optimizer.zero_grad()

        if args.single_modal:
            z1 = model(view_1.x, view_1.edge_index)[view_1.node_label_index]
            z2 = model(view_2.x, view_2.edge_index)[view_2.node_label_index]



        proj_z1 = model.projection(z1)
        proj_z2 = model.projection(z2)


        if args.self_tp:
            principal_component = all_principal_component[ori_graph.raw_nodes]
        else:
            principal_component = all_principal_component

        if args.use_tp:
            loss, ins_loss, contrast_loss = criterion(z1, z2, proj_z1, proj_z2, principal_component)
            total_ins_loss += ins_loss * proj_z1.shape[0]
            total_con_loss += contrast_loss * proj_z1.shape[0]
            total_loss += (ins_loss + contrast_loss) * proj_z1.shape[0]

        else:
            loss = criterion(proj_z1, proj_z2)
            total_loss += loss.data.item() * proj_z1.shape[0]



        loss.backward(retain_graph=True)

        optimizer.step()

        if step % args.log_every == 0:
            if args.use_tp:
                if args.with_multi_modal_loss:
                    pass
                else:
                    print('Step {:05d} | Contrastive Loss {:.4f}'.format(step,
                                                                                                              loss.item(),
                                                                                                              ins_loss,
                                                                                                              contrast_loss))

            else:
                print('Step {:05d} | Loss {:.4f}'.format(step, loss.item()))
        pbar.update()
    pbar.close()
    total_mean_loss = total_loss / train_id.shape[0]
    total_mean_instance_loss = total_ins_loss / train_id.shape[0]
    total_mean_contrastive_loss = total_con_loss / train_id.shape[0]
    if args.with_multi_modal_loss:
       pass
    else:
        print(
            f"Contrastive Loss: {total_mean_contrastive_loss}\n"
        )
        # swanlab.log({"Mean Test Loss":total_mean_loss, "Instance Loss":total_mean_instance_loss, "Contrastive Loss":total_mean_contrastive_loss})
        with open(args.result_save_path, "a", encoding="utf-8") as f:
            f.write(f"Contrastive Loss: {total_mean_contrastive_loss}\n")

    return total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss


@torch.no_grad()
def test(args, test_data, num_nodes):
    test_loader = NodeNegativeLoader(
        test_data,
        batch_size=512,
        shuffle=False,
        neg_ratio=0,
        num_neighbors=[-1],
        mask_feat_ratio_1=args.drop_feature_rate_1,
        mask_feat_ratio_2=args.drop_feature_rate_2,
        drop_edge_ratio_1=args.drop_edge_rate_1,
        drop_edge_ratio_2=args.drop_edge_rate_2,
    )

    model.eval()
    total_loss = 0
    total_ins_loss = 0
    total_con_loss = 0

    pbar = tqdm(total=len(test_loader))
    for step, (ori_graph, view_1, view_2) in enumerate(test_loader):
        ori_graph, view_1, view_2 = ori_graph.to(device), view_1.to(device), view_2.to(device)

        z1 = model(view_1.x, view_1.edge_index)[view_1.node_label_index]
        z2 = model(view_2.x, view_2.edge_index)[view_2.node_label_index]

        proj_z1 = model.projection(z1)
        proj_z2 = model.projection(z2)

        if args.self_tp:
            principal_component = all_principal_component[ori_graph.raw_nodes]
        else:
            principal_component = all_principal_component

        if args.use_tp:
            loss, ins_loss, contrast_loss = criterion(z1, z2, proj_z1, proj_z2, principal_component)
            total_ins_loss += ins_loss * proj_z1.shape[0]
            total_con_loss += contrast_loss * proj_z1.shape[0]
            total_loss += (ins_loss + contrast_loss) * proj_z1.shape[0]
        else:
            loss = criterion(proj_z1, proj_z2)
            total_loss += loss.data.item() * proj_z1.shape[0]
        pbar.update()
    pbar.close()

    total_mean_loss = total_loss / num_nodes
    total_mean_instance_loss = total_ins_loss / num_nodes
    total_mean_contrastive_loss = total_con_loss / num_nodes

    print(
        f"Mean Test Contrastive Loss: {total_mean_contrastive_loss}")

    return total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='cuda', help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='icews_wiki')
    argparser.add_argument('--dataset_path', type=str, default='data')
    argparser.add_argument('--result_save_path', type=str, default='/')
    argparser.add_argument('--data_num', type=str, default='all')
    argparser.add_argument('--with_multi_modal', default=False, action='store_true')
    argparser.add_argument('--with_multi_modal_loss', default=False, action='store_true')
    argparser.add_argument('--single_modal', default=False, action='store_true')
    argparser.add_argument('--two_modal', default=False, action='store_true')
    argparser.add_argument('--fusion', type=str, default='original', choices=['original', 'attention'])
    argparser.add_argument('--num_epochs', type=int, default=20)
    argparser.add_argument('--num_runs', type=int, default=1)
    argparser.add_argument('--num_hidden', type=int, default=300)
    argparser.add_argument('--num_out', type=int, default=4096)
    argparser.add_argument('--num_layers', type=int, default=2)
    argparser.add_argument('--num_negs', type=int, default=0)
    argparser.add_argument('--patience', type=int, default=10)
    argparser.add_argument('--fan_out', type=str, default='25,10')
    argparser.add_argument('--batch_size', type=int, default=256)
    # argparser.add_argument(
    #     "--swanlab",
    #     action="store_false",
    #     default=True,
    #     help="Whether to start Swanlab for log"
    # )
    # argparser.add_argument('--log_every', type=int, default=20)
    argparser.add_argument('--log_every', type=int, default=10)
    argparser.add_argument('--eval_every', type=int, default=50)
    argparser.add_argument('--lr', type=float, default=0.002)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num_workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--lazy_load', default=True)
    argparser.add_argument('--use_tp', default=True)
    argparser.add_argument('--self_tp', default=False)
    argparser.add_argument('--drop_edge_rate_1', type=int, default=0.3)
    argparser.add_argument('--drop_edge_rate_2', type=int, default=0.4)
    argparser.add_argument('--drop_feature_rate_1', type=int, default=0.0)
    argparser.add_argument('--drop_feature_rate_2', type=int, default=0.1)
    argparser.add_argument('--tau', type=int, default=0.4)
    argparser.add_argument('--gnn_type', type=str, default='sage')
    argparser.add_argument('--pretrain_gnn', type=str, default='pretrain_gnn')


    args = argparser.parse_args()

    fans_out = [int(i) for i in args.fan_out.split(',')]
    assert len(fans_out) == args.num_layers

    # if args.gpu >= 0:
    if args.gpu == 'mps' or args.gpu == 'cuda':
        print("Using GPU {}".format(args.gpu))
        device = torch.device(args.gpu)
    else:
        print("Using CPU {}".format(args.gpu))
        device = torch.device('cpu')

    if not os.path.exists(f'./saved_model/gnn'):
        os.makedirs(f'./saved_model/gnn')

    path = osp.join('./data', 'kg', 'kg_' + args.dataset + '_data.pt')
    datasets = torch.load(path)
    # 选择一个具体图（例如 FB15K_DB15K）
    x = datasets[args.dataset]['x']
    edge_index = datasets[args.dataset]['edge_index']
    attr_feat = datasets[args.dataset]['attr_feat']
    vis_feat = datasets[args.dataset]['vis_feat']
    rel_feat = None

    # 构建 PyG Data 对象（保持与原代码结构一致）
    data_list = []
    data = torch_geometric.data.Data(x=x, edge_index=edge_index)
    data.attr_feat = attr_feat
    data.vis_feat = vis_feat
    # data.rel_feat = None
    data_list.append(data)

    # 保持原来的循环结构（仅一项）
    n_ls = []
    for data in data_list:
        n_d = torch_geometric.transforms.ToUndirected()(data)
        n_d = torch_geometric.transforms.AddRemainingSelfLoops()(n_d)
        n_ls.append(n_d)

    data = n_ls[0]

    # ✅ 类型转换（与原相同）
    # data.x = data.x.type(torch.float32)
    if attr_feat is not None:
        data.attr_feat = attr_feat.type(torch.float32)
    else:
        data.attr_feat = None

    if vis_feat is not None:
        data.vis_feat = vis_feat.type(torch.float32)
    else:
        data.vis_feat = None


    # ✅ train_id 改成节点数量对应的索引
    train_id = data.x

    print(f"Loaded KG {args.dataset}:  #Entities={data.num_nodes}")
    if not os.path.exists(f"./results/{args.dataset}"):
        os.makedirs(f"./results/{args.dataset}")
    with open(args.result_save_path, "a", encoding="utf-8") as f:
        f.write(f"Loaded KG {args.dataset}:  #Entities={data.num_nodes}\n")


    # 使用 Xavier / Glorot 初始化
    def xavier_init(num_nodes, feat_dim, device=device):
        tensor = torch.empty(num_nodes, feat_dim, device=device)
        return nn.init.xavier_uniform_(tensor)


    def get_embeddings(file_dir, img_path, attr_emb_path_1, attr_emb_path_2, e, topR=1000, lang_list=[1, 2]):
        ent2id_dict, ills, triples, r_hs, r_ts, ids = (
            read_raw_data(file_dir, lang_list)
        )

        attr_1 = json.load(open(attr_emb_path_1, "r"))
        attr_2 = json.load(open(attr_emb_path_2, "r"))

        attr_dict = {**attr_1, **attr_2}
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

        return torch.tensor(img_embeddings, dtype=torch.float32, device=device), torch.tensor(attr_embeddings,
                                                                                              dtype=torch.float32,
                                                                                              device=device), None



    path = args.dataset_path + args.dataset + "/"
    img_path = path + args.dataset + "_id_img_feature_dict.pkl"
    attr_path1 = path + "attr_summary_emb_1.json"
    attr_path2 = path + "attr_summary_emb_2.json"


    if args.single_modal:
        print("Single-modal：transe实体嵌入")
        initial_embeddings = torch.load(f"./data/kg/{args.dataset}_initial_embeddings.pt", map_location=device)
        initial_embeddings = initial_embeddings.detach().clone()
        data.x = initial_embeddings

        data.attr_feat = None
        data.vis_feat = None
        data.rel_feat = None

    # ✅ train_id 改成节点数量对应的索引
    train_id = data.x
    for run in range(args.num_runs):
        seed_everything(run)

        data = data.to(device, 'x', 'edge_index')

        num_node_features = data.x.shape[1]

        print("entity feature dim:", num_node_features)

        if args.single_modal:
            model = GraphSAGE(
                num_node_features,
                hidden_channels=args.num_hidden,
                out_channels=args.num_out,
                n_layers=args.num_layers,
                num_proj_hidden=args.num_out,
                activation=F.relu,
                dropout=args.dropout,
                edge_dim=None,
                gnn_type=args.gnn_type
            )

        model = model.to(dtype=torch.float32, device=device)
        print(model)

        all_principal_component = torch.load('./data/llm/llm_pca.pt').to(device, dtype=torch.float32)

        if args.use_tp:
            criterion = ContrastiveLoss(args.tau, self_tp=args.self_tp).to(device)
        else:
            criterion = GraceLoss(args.tau).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        no_increase = 0
        best_loss = 1000000000
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch}")
            total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss = train(args, data, model, optimizer,
                                                                                           criterion)
            if total_mean_loss < best_loss:
                best_loss = total_mean_loss
                no_increase = 0

                if args.single_modal:
                    torch.save(model.state_dict(),
                               f'./saved_model/gnn/{args.pretrain_gnn}_{args.dataset}.pth')
                    print("保存")

            else:
                no_increase += 1
                if no_increase > args.patience:
                    break
