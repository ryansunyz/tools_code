import matplotlib.pyplot as plt


def Gradient_Ascent_test():
    """梯度上升算法 吃函数
    f(x)=-x^2+4x的最大值
    :return:
    """

    def f(x):
        return -x ** 2 + 4

    def f_prime(x_old):
        return -2 * x_old + 4  # f(x)的倒数

    y_list = []
    i = 0
    x_index = []
    x_old = -1
    x_new = 0
    alpha = 0.01  # 步长
    presision = 0.00000001  # 更新阈值
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * f_prime(x_old)
        y_list.append(x_new)
        x_index.append(i)
        i += 1
    plt.plot(x_index, y_list)
    plt.show()


if __name__ == '__main__':
    Gradient_Ascent_test()
