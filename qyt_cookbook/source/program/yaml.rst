==================
YAML
==================

- YAML 是 "YAML Ain't a Markup Language"（YAML 不是一种标记语言）的递归缩写。在开发的这种语言时，YAML 的意思其实是："Yet Another Markup Language"（仍是一种标记语言）。
- YAML 的语法和其他高级语言类似，并且可以简单表达清单、散列表，标量等数据形态。它使用空白符号缩进和大量依赖外观的特色，特别适合用来表达或编辑数据结构、各种配置文件、倾印调试内容、文件大纲（例如：许多电子邮件标题格式和YAML非常接近）。
- YAML 的配置文件后缀为: ``.yml``，如：``runoob.yml`` 。
- YAML菜鸟教程： https://www.runoob.com/w3cnote/yaml-intro.html
- YAML与json互转： https://qqe2.com/json/jsonyaml
- 重要语法： 

    - **大小写敏感**
    - **使用缩进表示层级关系，缩进不允许使用tab，只允许空格**
    - 缩进的空格数不重要，只要相同层级的元素左对齐即可

基本数据结构
######################

.. code-block:: shell

    boolean: 
        - TRUE  # true,True都可以
        - FALSE  # false，False都可以
    float:
        - 3.14
        - 6.8523015e+5  # 可以使用科学计数法
    int:
        - 123
        - 0b1010_0111_0100_1010_1110    # 二进制表示
    null:
        nodeName: 'node'
        parent: ~  # 使用~表示null
    string:
        - 哈哈
        - 'Hello world'  # 可以使用双引号或者单引号包裹特殊字符
        - newline
          newline2    # 字符串可以拆成多行，每一行会被转化成一个空格
    date:
        - 2018-02-17    # 日期必须使用ISO 8601格式，即yyyy-MM-dd
    datetime: 
        -  2018-02-17T15:02:31+08:00    # 时间使用ISO 8601格式，时间和日期之间使用T连接，最后使用+代表时区

锚点与引用
######################

- 符号意义： ``&`` 用来建立锚点（defaults），``<<`` 表示合并到当前数据，``*`` 用来引用锚点。
- 锚点与引用yml文件格式：

.. code-block:: shell

    efaults: &defaults  # 建立锚点
      adapter:  postgres
      host:     localhost
    development:
      database: myapp_development
      aa: *defaults  # 引用锚点
    test:
      database: myapp_test
      <<: *defaults  # 合并到当前数据

- 对应转换后的json数据格式：

.. code-block:: shell

    {
      "efaults": {
        "adapter": "postgres",
        "host": "localhost"
      },
      "development": {
        "database": "myapp_development",
        "aa": {
          "adapter": "postgres",
          "host": "localhost"
        }
      },
      "test": {
        "database": "myapp_test",
        "adapter": "postgres",
        "host": "localhost"
      }
    }