import sys
import re
import math
from collections import Counter

def text_preprocess(raw_text: str) -> str:
    """
    文本预处理：清除所有非中文、英文、数字字符
    :param raw_text: 原始读取文本
    :return: 清洗后纯文本
    """
    # 正则保留中文、大小写字母、数字，其余全部剔除
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', raw_text)
    return cleaned


def simple_tokenize(clean_text: str) -> list[str]:
    """
    分词函数：英文/数字作为单词，中文按单字切分
    :param clean_text: 预处理后的干净文本
    :return: token列表
    """
    tokens = []
    # 提取英文、数字串
    en_pattern = re.compile(r'[a-zA-Z0-9]+')
    en_matches = en_pattern.findall(clean_text)
    for word in en_matches:
        tokens.append(word)
        clean_text = clean_text.replace(word, "", 1)
    # 剩余中文逐个切分
    for ch in clean_text:
        tokens.append(ch)
    return tokens


def calc_cosine_similarity(vec1: Counter, vec2: Counter) -> float:
    """
    核心计算模块：余弦相似度计算
    :param vec1: 原文词频向量Counter
    :param vec2: 抄袭文本词频向量Counter
    :return: 相似度 0~1
    """
    all_keys = set(vec1.keys()).union(vec2.keys())
    dot_product = 0.0
    mag1 = 0.0
    mag2 = 0.0

    for key in all_keys:
        v1 = vec1.get(key, 0)
        v2 = vec2.get(key, 0)
        dot_product += v1 * v2
        mag1 += v1 ** 2
        mag2 += v2 ** 2

    # 边界：空文本，直接返回0，防止除0异常
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (math.sqrt(mag1) * math.sqrt(mag2))


def read_text_file(file_path: str) -> str:
    """
    读取文本文件，优先utf-8，失败自动切换gbk，兼容Windows记事本中文文件
    :param file_path: 文件绝对路径
    :return: 文件全部文本字符串
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        with open(file_path, 'r', encoding='gbk') as f:
            return f.read()


def main():
    # 命令行参数校验，必须3个文件路径参数
    if len(sys.argv) != 4:
        print("参数错误！使用方式：python main.py 原文路径 抄袭文本路径 输出答案路径")
        return

    orig_path, add_path, ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # 读取两份文本
    orig_text = read_text_file(orig_path)
    add_text = read_text_file(add_path)

    # 预处理
    clean_orig = text_preprocess(orig_text)
    clean_add = text_preprocess(add_text)

    # 分词
    tokens_orig = simple_tokenize(clean_orig)
    tokens_add = simple_tokenize(clean_add)

    # 构建词频向量
    vec_orig = Counter(tokens_orig)
    vec_add = Counter(tokens_add)

    # 计算相似度
    sim = calc_cosine_similarity(vec_orig, vec_add)
    sim_percent = round(sim * 100, 2)

    # 写入答案文件，保留2位小数
    with open(ans_path, 'w', encoding="utf-8") as out_f:
        out_f.write(f"{sim_percent:.2f}")
    print(f"计算完成，重复率：{sim_percent:.2f}%，结果写入{ans_path}")


if __name__ == "__main__":
    main()


