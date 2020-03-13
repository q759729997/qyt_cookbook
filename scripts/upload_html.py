"""
使用pscp实现Windows 和 Linux服务器间远程传递文件:https://blog.csdn.net/sgmcumt/article/details/79135395
Python实现Windows和Linux之间互相传输文件(文件夹)的方法:http://www.cppcns.com/os/linux/190042.html
"""
import os
import sys
sys.path.insert(0, './')
from scripts.myconfig import upload_config  # noqa


def win_to_linux_file(win_path, linux_path, linux_ip, user_name, password):
    print("begin upload")
    cmd = r'D:\\qiao\\pscp.exe -pw {} -r {} {}@{}:{}'.format(password, win_path, user_name, linux_ip, linux_path)
    print('cmd:{}'.format(cmd))
    os.system(cmd)


if __name__ == "__main__":
    win_path = 'D:\\workspace\\github_qyt\\qyt_cookbook\\qyt_cookbook\\build\\html\\*'
    linux_path = '/var/www/html/qyt_cookbook/'
    linux_ip = upload_config['linux_ip']
    user_name = upload_config['user_name']
    password = upload_config['password']
    win_to_linux_file(win_path, linux_path, linux_ip, user_name, password)
