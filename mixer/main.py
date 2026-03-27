import argparse
import datetime
import os.path
import sys
from collections import Counter
import torch
import json
import numpy as np
import swanlab# 导入SwanLab Python库


def main(args):
    # 主流程
    # 先判断操作系统
    if sys.platform.startswith('linux'):
        # Linux 服务器环境
        base_path = "data/"

        print(f"检测到 Linux 系统，使用路径: {base_path}")
    else:
        # 非 Linux 环境（如 macOS、Windows）
        base_path = "data/mmkb-datasets/"
        print(f"检测到 {sys.platform} 系统，使用路径: {base_path}")
    if "_en" in args.dataset:
        # 注意：这里也要根据系统调整路径前缀
        if sys.platform.startswith('linux'):
            base_path = "data/"
        else:
            base_path = "data/DBP15K/"

    triples_path_1 = base_path + args.dataset + "/triples_1"
    triples_path_2 = base_path + args.dataset + "/triples_2"
    attr_emb_path_1 = base_path + args.dataset + "/attr_summary_emb_1.json"
    attr_emb_path_2 = base_path + args.dataset + "/attr_summary_emb_2.json"

    # 1. 检测可用的设备
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        print("Using Apple Silicon MPS")

    def load_triples_and_counts(triples_path_1, triples_path_2):
        """加载三元组，并返回正确的实体、关系计数及三元组列表"""
        triples = []
        max_entity_id = 0
        max_relation_id = 0

        # 加载第一个文件
        with open(triples_path_1, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                h, r, t = parts[0], parts[1], parts[2]
                h, r, t = int(h), int(r), int(t)
                triples.append((h, r, t))
                max_entity_id = max(max_entity_id, h, t)
                max_relation_id = max(max_relation_id, r)

        # 加载第二个文件
        with open(triples_path_2, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                h, r, t = parts[0], parts[1], parts[2]
                h, r, t = int(h), int(r), int(t)
                triples.append((h, r, t))
                max_entity_id = max(max_entity_id, h, t)
                max_relation_id = max(max_relation_id, r)

        # Embedding层需要：num_entities = max_entity_id + 1
        #                 num_relations = max_relation_id + 1
        num_entities = max_entity_id + 1
        num_relations = max_relation_id + 1

        print(f"三元组数: {len(triples)}")
        print(f"最大实体ID: {max_entity_id} -> 实体总数: {num_entities}")
        print(f"最大关系ID: {max_relation_id} -> 关系总数: {num_relations}")

        return triples, num_entities, num_relations

    # 1. 加载三元组并获取正确的计数
    triples, num_entities, num_relations = load_triples_and_counts(triples_path_1, triples_path_2)
    print(f"加载三元组: {len(triples)} 个")

    # 2. 统计实体和关系数量
    print(f"实体数: {num_entities}, 关系数: {num_relations}")

    # 3. 加载属性嵌入
    attr_1 = json.load(open(attr_emb_path_1, "r"))
    attr_2 = json.load(open(attr_emb_path_2, "r"))
    attr_dict = {**attr_1, **attr_2}  # 合并
    length_counter = Counter()
    for eid, emb in attr_dict.items():
        if isinstance(emb, (list, tuple, np.ndarray)):
            length_counter[len(emb)] += 1
        else:
            length_counter['scalar_or_other'] += 1
    print("Embedding length distribution in attr_dict:")
    for k, v in length_counter.items():
        attr_dim = k
        print(f"Length {k}: {v} entities")

    # 将属性嵌入矩阵移到设备上
    attr_matrix = torch.zeros((num_entities, attr_dim), dtype=torch.float32, device=device)

    for entity_id_str, embedding in attr_dict.items():
        entity_id = int(entity_id_str)  # 假设键是字符串形式的实体ID
        if entity_id < num_entities:
            # 先将数据加载到CPU，然后移到设备
            attr_matrix[entity_id] = torch.tensor(embedding, dtype=torch.float32).to(device)

    print(f"属性嵌入矩阵形状: {attr_matrix.shape}")

    # 5. 初始化并训练TransE模型
    from model import KnowledgeGraphEnhancedEmbedding

    kge_system = KnowledgeGraphEnhancedEmbedding(
        num_entities=num_entities,
        num_relations=num_relations,
        attr_dim=attr_dim,
        rel_dim=num_relations,
        hidden_dim=args.hidden_dim,
        fusion_type='gated',
        device=device  # 新增：传递设备参数
    )
    # 放在 train_relation_embeddings 调用之前
    print(f"TransE模型定义的关系数: {kge_system.transe_model.num_relations}")
    print(f"实际三元组中的最大关系ID: {max(r for _, r, _ in triples)}")

    # 6. 训练TransE（修改train_relation_embeddings以支持设备）
    rel_embeddings = kge_system.train_relation_embeddings(
        triples=triples,
        epochs=args.transe_epochs,
        batch_size=args.batch_size,
        device=device  # 新增：传递设备参数
    )

    # 7. 获取所有实体的融合嵌入
    entity_ids = list(range(num_entities))
    # 注意：entity_ids需要放在设备上（如果是MPS，需要额外处理）
    entity_ids_tensor = torch.tensor(entity_ids, dtype=torch.long).to(device)
    initial_embeddings = kge_system.get_initial_embeddings(
        entity_ids=entity_ids_tensor,  # 使用张量形式
        llm_attr_embeddings=attr_matrix[entity_ids_tensor]
    )
    print(f"初始嵌入设备: {initial_embeddings.device}")
    initial_embeddings = initial_embeddings.cpu()
    torch.save(initial_embeddings, f"./data/kg/{args.dataset}_initial_embeddings.pt")
    print(f"融合后的初始嵌入形状: {initial_embeddings.shape}")
    print(f"保存到./data/kg/{args.dataset}_initial_embeddings.pt")
    print("完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--swanlab",
        action="store_false",
        default=True,
        help="Whether to start Swanlab for log"
    )
    parser.add_argument(
        "--experiment_name",
        default="",
        type=str,
        help="experiment_name"
    )
    parser.add_argument(
        "--hidden_dim",
        default=2560,
        type=int,
        help="hidden_dim"
    )
    parser.add_argument(
        "--transe_epochs",
        default=1,
        type=int,
        help="epochs"
    )
    parser.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        help="batch_size"
    )
    parser.add_argument(
        "--dataset",
        default="icews_yago",
        help="dataset"
    )
    args = parser.parse_args()

    # 获取当前时间
    current_datetime = datetime.datetime.now()

    # 格式化日期和时间为字符串
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")

    # 开启一个SwanLab实验
    # run = swanlab.init(
    #     project="EA-LLM",
    #     experiment_name=args.dataset + args.experiment_name + formatted_datetime,
    #     description="",
    #     config=args
    # )

    if not os.path.exists(f"./data/kg/{args.dataset}_initial_embeddings.pt"):
        print(f"./data/kg/{args.dataset}_initial_embeddings.pt")
        print("transe未训练")
        main(args)