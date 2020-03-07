"""
    main_module - rst文件中图片路径规范化处理.

    Main members:

        # __main__ - 主函数入口.
"""
import codecs

from qytPython.tools.file import get_file_names_recursion
from qytPython.tools.file import read_file_texts


img_file_prefix = 'D:\\workspace\\github_qyt\\qyt_cookbook\\qyt_cookbook\\source\\'


def pick_need_to_process_rst_files(rst_files):
    """ 选出包含图片且尚未规范化处理的文件.

        @params:
            rst_files - rst文件.

        @return:
            On success - 待处理的文件.
            On failure - 错误信息.
    """
    need_to_process_rst_files = list()
    for file_name in rst_files:
        file_texts = read_file_texts(file_name)
        for text in file_texts:
            if img_file_prefix in text:
                need_to_process_rst_files.append(file_name)
                break
    return need_to_process_rst_files


if __name__ == "__main__":
    rst_path = './qyt_cookbook/source'
    file_names = list()
    get_file_names_recursion(rst_path, file_names)
    print('file_names len:{}'.format(len(file_names)))
    # 提取rst文件
    rst_files = [file_name for file_name in file_names if file_name.endswith('.rst')]
    print('rst_files len:{}'.format(len(rst_files)))
    # 提取待处理的文件
    need_to_process_rst_files = pick_need_to_process_rst_files(rst_files)
    print('need_to_process_rst_files len:{}'.format(len(need_to_process_rst_files)))
    print(need_to_process_rst_files[:5])
    # 开始进行处理
    for file_name in need_to_process_rst_files:
        print('process file:{}'.format(file_name))
        file_texts = read_file_texts(file_name, keep_original=True)
        with codecs.open(file_name, mode='w', encoding='utf8') as fw:
            for text in file_texts:
                if img_file_prefix in text:
                    print(text)
                    text = text.replace('\\', '/')
                    print(text)
                fw.write(text)
