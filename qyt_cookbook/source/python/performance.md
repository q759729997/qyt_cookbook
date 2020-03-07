# 性能测试

## 运行时间评测

- 参考资料，Python性能分析大全：<https://selfboot.cn/2016/06/13/python_performance_analysis/>
- python提供了模块 time，其中 time.clock() 在Unix/Linux下返回的是CPU时间(浮点数表示的秒数)，Win下返回的是以秒为单位的真实时间(Wall-clock time)。

~~~python
# 待测试程序
import time

def time_test():
    for i in range(10):
        time.sleep(0.1)

# 测试脚本
def _time_analyze_(func):
    from time import clock
    start = clock()
    func()
    finish = clock()
    print("{:<20}{:10.6} s".format(func.__name__ + ":", finish - start))

_time_analyze_(time_test)  # time_test:             1.00729 s
~~~

- Python 提供了timeit模块，用来测试代码块的运行时间。

~~~python
def _timeit_analyze_(func):
    from timeit import Timer
    t1 = Timer("%s()" % func.__name__, "from __main__ import %s" % func.__name__)
    print("{:<20}{:10.6} s".format(func.__name__ + ":", t1.timeit(1)))

_time_analyze_(time_test)  # time_test:             1.00695 s
~~~

- 复杂情况运行时间测试，参考上述链接内的`Profile`。