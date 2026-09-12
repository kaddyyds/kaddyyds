from collections import Counter
from main import text_preprocess, simple_tokenize, calc_cosine_similarity, read_text_file
import os

# 测试用例 1：文本预处理-去除标点符号
def test_preprocess_punctuation():
    raw = "今天是星期天，天气晴！晚上看电影。abc123"
    res = text_preprocess(raw)
    assert res == "今天是星期天天气晴晚上看电影abc123"

# 测试用例 2：预处理-全标点空文本
def test_preprocess_all_symbol():
    raw = "！，。？；："
    res = text_preprocess(raw)
    assert res == ""

# 测试用例3：分词-中英文混合
def test_token_mix():
    text = "python编程"
    tokens = simple_tokenize(text)
    assert "python" in tokens
    assert "编" in tokens
    assert "程" in tokens

# 测试用例4：分词-纯中文
def test_token_only_cn():
    text = "今天晴天"
    tokens = simple_tokenize(text)
    assert tokens == ["今","天","晴","天"]

# 测试用例5：余弦完全相同文本，相似度=1
def test_cosine_same():
    v1 = Counter(["今","天","晴"])
    v2 = Counter(["今","天","晴"])
    sim = calc_cosine_similarity(v1, v2)
    assert abs(sim - 1.0) < 0.001

# 测试用例6：余弦完全无关文本，相似度0
def test_cosine_norelate():
    v1 = Counter(["苹果"])
    v2 = Counter(["电脑"])
    sim = calc_cosine_similarity(v1, v2)
    assert abs(sim - 0.0) < 0.001

# 测试用例7：其中一篇为空文本，返回0
def test_cosine_one_empty():
    v1 = Counter()
    v2 = Counter(["今","天"])
    sim = calc_cosine_similarity(v1, v2)
    assert sim == 0.0

# 测试用例8：部分相似文本，验证中间值
def test_cosine_partial():
    v1 = Counter(["a","b","c"])
    v2 = Counter(["a","b","d"])
    sim = calc_cosine_similarity(v1, v2)
    assert 0 < sim < 1

# 测试用例9：文件读取utf8
def test_read_utf8():
    tmp_name = "tmp_utf8.txt"
    with open(tmp_name,"w",encoding="utf-8") as f:
        f.write("测试文本")
    content = read_text_file(tmp_name)
    os.remove(tmp_name)
    assert content == "测试文本"

# 测试用例10：文件读取gbk编码
def test_read_gbk():
    tmp_name = "tmp_gbk.txt"
    with open(tmp_name,"w",encoding="gbk") as f:
        f.write("中文gbk")
    content = read_text_file(tmp_name)
    os.remove(tmp_name)
    assert content == "中文gbk"

if __name__ == "__main__":
    test_preprocess_punctuation()
    test_preprocess_all_symbol()
    test_token_mix()
    test_token_only_cn()
    test_cosine_same()
    test_cosine_norelate()
    test_cosine_one_empty()
    test_cosine_partial()
    test_read_utf8()
    test_read_gbk()
    print("✅ 全部10个单元测试用例执行通过！")

