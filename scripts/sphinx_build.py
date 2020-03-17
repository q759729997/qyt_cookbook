import os

if __name__ == "__main__":
    """
    设置sphinx-build.exe环境变量：https://blog.csdn.net/Astilbe001/article/details/104413087
    D:\\ProgramData\\Miniconda3\\envs\\python36\\Scripts\\ 添加到环境变量path
    重启vscode后执行
    """
    os.system(r'D:\workspace\github_qyt\qyt_cookbook\qyt_cookbook\make.bat clean')
    os.system(r'D:\workspace\github_qyt\qyt_cookbook\qyt_cookbook\make.bat html')
