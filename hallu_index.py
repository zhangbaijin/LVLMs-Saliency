import torch

def find_sentence_index(new_tokens, hallu_word, tokenizer):
    """在生成的 token 序列中查找幻觉词的起始索引。"""
    # 将幻觉词编码为 token 序列（不添加特殊符号）
    hallu_ids = tokenizer.encode(hallu_word, add_special_tokens=False)
    #print("测试输出：")
    #print(f"new_tokens: {new_tokens}")
    #print(f"hallu_ids: {hallu_ids}")
    #print(f"tokenizer.decode(new_tokens):{tokenizer.decode(new_tokens)}")
    for i in range(len(new_tokens) - len(hallu_ids) + 1):
        if new_tokens[i:i+len(hallu_ids)] == hallu_ids:
            return i
    return -1  # 未找到幻觉词序列

def compute_hallu_output(new_tokens, hallu_word, tokenizer):
    word = hallu_word
    # 移除末尾的特殊结束符（如 Qwen2 的<|endoftext|>），确保输出不包含多余符号
    if tokenizer.eos_token_id is not None and new_tokens.numel() > 0 and new_tokens[-1].item() == tokenizer.eos_token_id:
        new_tokens = new_tokens[:-1]

    
    # 1. 查找幻觉词起始位置索引
    # 将 new_tokens 转为一维 Python 列表，便于后续处理（如果是tensor，需要 .tolist()）
    new_tokens_list = new_tokens.tolist() if isinstance(new_tokens, torch.Tensor) else list(new_tokens)
    sentence_index = find_sentence_index(new_tokens_list, word, tokenizer)
    
    # 2. 获取幻觉词的词表ID（取第一个token的ID）
    hallu_token_ids = tokenizer.encode(word, add_special_tokens=False)
    if len(hallu_token_ids) == 0:
        raise ValueError(f"无法编码幻觉词: {hallu_word}")
    hallu_logits_index = hallu_token_ids[0]
    
    # 3. 截取包含幻觉词的输出子文本（tokens）并解码
    #end_index = sentence_index + len(hallu_token_ids)
    #out_sub_tokens = new_tokens[:end_index]
    out_sub_tokens = new_tokens[:sentence_index]
    out_sub_text = tokenizer.decode(out_sub_tokens, skip_special_tokens=True)
    # 解码子文本，跳过特殊符号以避免包含<|endoftext|>等
    out_sub_text = tokenizer.decode(out_sub_tokens, skip_special_tokens=True)
    
    # 4. 解码完整输出文本，同样跳过特殊符号
    #output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return  sentence_index , hallu_logits_index, out_sub_text

