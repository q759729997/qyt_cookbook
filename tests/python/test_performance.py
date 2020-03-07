"""
    main_module - Python性能测试，测试时将对应方法的@unittest.skip注释掉.

    Main members:

        # __main__ - 程序入口.
"""
import unittest


class TestPerformance(unittest.TestCase):
    """Python性能测试.

    Main methods:
        test_calculate_time_consuming - 计算耗时.
    """
    # @unittest.skip('debug')
    def test_calculate_time_consuming(self):
        """计算耗时.
        """
        print('{} test_calculate_time_consuming {}'.format('-'*15, '-'*15))
        import time

        def time_test():
            for i in range(10):
                time.sleep(0.1)

        print('{} time.clock {}'.format('-'*15, '-'*15))

        def _time_analyze_(func):
            # python提供了模块 time，其中 time.clock() 在Unix/Linux下返回的是CPU时间(浮点数表示的秒数)，Win下返回的是以秒为单位的真实时间(Wall-clock time)。
            from time import clock
            start = clock()
            func()
            finish = clock()
            print("{:<20}{:10.6} s".format(func.__name__ + ":", finish - start))

        _time_analyze_(time_test)  # time_test:             1.00729 s

        print('{} timeit {}'.format('-'*15, '-'*15))

        def _timeit_analyze_(func):
            # Python 提供了timeit模块，用来测试代码块的运行时间
            from timeit import Timer
            t1 = Timer("%s()" % func.__name__, "from __main__ import %s" % func.__name__)
            print("{:<20}{:10.6} s".format(func.__name__ + ":", t1.timeit(1)))

        _time_analyze_(time_test)  # time_test:             1.00695 s


if __name__ == "__main__":
    unittest.main()  # 运行当前源文件中的所有测试用例
