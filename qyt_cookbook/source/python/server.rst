==================
Python服务器
==================

tornado
######################

- 使用手册： http://demo.pythoner.com/itt2zh/ch1.html#ch1-1-1
- github地址： https://github.com/tornadoweb/tornado

Flask
######################

- 使用手册： https://dormousehole.readthedocs.io/en/latest/quickstart.html#quickstart
- w3c使用手册： https://www.w3cschool.cn/flask/flask_overview.html

跨域
***************************

- 安装 ``flask_cors``

.. code-block:: python

    from flask_cors import CORS
    # r'/*' 是通配符，让本服务器所有的 URL 都允许跨域请求
    CORS(app, resources=r'/*')
    @app.route('/')
    def hello_world():
        return jsonify({'msg':"成功","erromsg":None})

Sanic
######################

- 使用手册： https://www.osgeo.cn/sanic/sanic/getting_started.html
- github地址： https://github.com/huge-success/sanic