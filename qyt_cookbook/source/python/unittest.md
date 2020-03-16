# 单元测试

## pytest

- 跳过某个测试用例：

~~~python
@pytest.mark.skip(reason="跳过原因")
~~~

- 运行时显示详细信息：

~~~python
if __name__ == '__main__':
    pytest.main(['-s', './test/import_time.py'])
~~~
