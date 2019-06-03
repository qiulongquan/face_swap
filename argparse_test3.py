# 这个是parameter参数分析器的程序
# python argparse_test3.py
# 0.01
# 2000
# 写入上面的命令会显示默认的parameter参数输出
# python argparse_test3.py --parameter1 3.231 --parameter2 11222
# 写入上面的命令会显示指定的parameter参数输出
# 3.231
# 11222

import argparse

# 基本モデル
FLAGS = None

def main():
    print(FLAGS.parameter1)
    print(FLAGS.parameter2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--parameter1',
        type=float,
        default=0.01,
        help='Parameter1.'
  )
    parser.add_argument(
        '--parameter2',
        type=int,
        default=2000,
        help='Parameter2.'
  )

FLAGS, unparsed = parser.parse_known_args()
main()