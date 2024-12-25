import torch  # 导入 PyTorch 库，用于张量计算和深度学习
import torch.nn.functional as F  # 导入 PyTorch 的神经网络功能模块，通常用于激活函数和其他功能

def select_kv(prefetch_idx, k_cache, v_cache):
    """选择并聚合Key的 KV 缓存，使用预测的索引

    在解码阶段，聚合与预测的预取索引对应的Key KV 缓存，使用嵌入函数。

    参数:
        prefetch_idx: 每个头和批次的Key KV 缓存令牌的索引 (n', 1, bh)
        k_cache: Key缓存 (n, bh, d)
        v_cache: 值缓存 (n, bh, d)

    返回:
        selected_k: 选定的键缓存 (n', bh, d)
        selected_v: 选定的值缓存 (n', bh, d)
    """

    # 将预取索引展平，去掉多余的维度，并将其移动到与 k_cache 相同的设备上
    prefetch_idx = prefetch_idx.squeeze().to(k_cache.device)
    
    # 计算索引：基础索引加上每个批次的头的范围
    ind = prefetch_idx * k_cache.shape[1] + torch.arange(k_cache.shape[1])[None, :]
    
    # 使用嵌入函数从Key缓存中选择特征，reshape 以适应嵌入格式
    selected_k = F.embedding(ind, k_cache.reshape(-1, k_cache.shape[2]))
    
    # 使用嵌入函数从值缓存中选择特征，reshape 以适应嵌入格式
    selected_v = F.embedding(ind, v_cache.reshape(-1, v_cache.shape[2]))
    
    # 返回选定的键和值缓存
    return selected_k, selected_v


def speculate_attention(hidden, p_w_q, p_k_c, n_head, alpha, max_num_kv):
    """推测下一层注意力机制的Key KV 缓存的索引。

    在解码阶段，使用隐藏状态（层 i）、部分查询权重（层 i+1），
    和部分Key缓存（层 i+1），推测下一层的注意力得分。
    之后计算Key令牌的数量，获取具有高注意力得分的 top-k KV 缓存令牌的索引。

    参数:
        hidden: 层 i 的隐藏状态 (b, 1, D)
        p_w_q: 部分查询权重 (D', D)
        p_k_c: 部分Key缓存 (n, bh, d')

        注意：bh * d' == D'

    返回:
        prefetch_idx: 每个头和批次的Key KV 缓存令牌的索引 (n', 1, bh)
    """
    
    # 获取批次的大小
    b = hidden.shape[0]
    
    # 通过线性变换计算部分查询向量
    p_q = F.linear(hidden, p_w_q, bias=None)
    
    # 调整 p_q 的形状以方便后续操作
    p_q = p_q.view(b, 1, n_head, -1)
    
    # 重新排列 p_q 的维度，使其符合后续的矩阵乘法要求
    p_q = p_q.permute(0, 2, 1, 3).reshape(b * n_head, 1, -1)

    # 计算注意力得分，通过批量矩阵乘法将查询和Key缓存进行匹配
    p_attn = torch.bmm(p_q, p_k_c.permute(1, 2, 0))
    
    # 找到每个批次中注意力得分的最大值
    max_ = torch.max(p_attn, dim=-1)[0]
    
    # 计算一个阈值，基于 alpha 的偏移量
    thr_ = (max_ - alpha).unsqueeze(-1).repeat(1, 1, p_attn.shape[-1])
    
    # 根据阈值创建一个计数张量，标记哪些得分超过了阈值
    count = torch.where(
        p_attn > thr_, torch.ones_like(p_attn), torch.zeros_like(p_attn)
    )
    
    # 计算平均值，表示超过阈值的Key令牌的数量
    mean = torch.mean(torch.sum(count, dim=-1)).item()
    
    # 使用 torch.topk 获取具有最高得分的前 k 个Key索引
    prefetch_idx = torch.topk(
        p_attn.permute(2, 1, 0), min(int(mean), max_num_kv), dim=0
    )[1]

    # 返回预取的索引
    return prefetch_idx