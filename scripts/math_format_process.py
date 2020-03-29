"""
    main_module - rst所需数学公式格式转换.

    Main members:

        # __main__ - 主函数入口.
"""
import codecs
from copy import deepcopy

from qytPython.tools.file import read_file_iter


if __name__ == "__main__":
    file_name = './data/math.txt'
    file_iter = read_file_iter(file_name)
    # 处理行间公式
    texts = list()
    math_start = True
    for row_text in file_iter:
        text = row_text.strip()
        if text == '$$':
            if math_start:
                texts.append('')
                texts.append('.. math::')
                texts.append('')
                math_start = False
            else:
                texts.append('')
                math_start = True
        else:
            if not math_start:
                texts.append('    {}'.format(text))
            else:
                texts.append(text)
    # print('\n'.join(texts))
    # 处理行间公式
    original_texts = deepcopy(texts)
    texts = list()
    math_start = True
    for text in original_texts:
        row_chars = list()
        for char in list(text):
            if char == '$':
                if math_start:
                    row_chars.append(' :math:`')
                    math_start = False
                else:
                    row_chars.append('` ')
                    math_start = True
            else:
                row_chars.append(char)
        texts.append(''.join(row_chars))
    print('\n'.join(texts))
    # 输出
    with codecs.open(file_name, mode='w', encoding='utf8') as fw:
        for text in texts:
            fw.write('{}\n'.format(text))
