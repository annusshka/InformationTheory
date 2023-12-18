"""
При записи каких-либо двумерных массивов необходимо придерживаться правилу:
    - первый элемент указывает на кол-во эл. массива
    - второй элемент указывает на кол-во подэлементов в каждом элюмассива
Например, p(xiyj) = np.array([[0.4, 0.2, 0.2], [0.1, 0.05, 0.05]])
          где j = 1..3, i = 1..2
"""

import math
import numpy as np

EPS = 1e-6
eps_sign = 7


def get_from_console_xy(len_x, len_y):
    print("Введите элементы массива каждый с новой строчки по порядку: x1y1 x1y2 ... xny1 ... xnyn")
    return [[float(input()) for _ in range(len_y)] for _ in range(len_x)]


def get_from_console_yx(len_x, len_y):
    return get_from_console_xy(len_y, len_x)


def get_prob_from_xy(xy, s1, s2):
    prob = []

    for i in range(len(xy)):
        s1 += f'p({s2}{i + 1}) = '
        s1 += " + ".join([str(num) for num in xy[i]])
        sum_ = round(sum(xy[i]), eps_sign)
        prob.append(sum_)
        s1 += f' = {sum_}\n'
    return prob, s1


def get_x_from_xy(xy):
    return get_prob_from_xy(xy, '\nВероятности событий ансамбля X\n', "x")


def get_y_from_xy(xy):
    return get_prob_from_xy(np.transpose(xy), '\nВероятности событий ансамбля Y\n', "y")


def is_independent(xy, x, y):
    for i in range(len(xy)):
        for j in range(len(xy[0])):
            if abs(xy[i][j] - x[i] * y[j]) > EPS:
                return False
    return True


def is_independent_text(xy, x, y):
    s = "\nПроверка на независимость ансамблей:\n"
    if is_independent(xy, x, y):
        s += "Ансамбли независимы\n"
    else:
        s += "Ансамбли зависимы\n"
    for i in range(len(xy)):
        for j in range(len(xy[0])):
            s += f'p(x{i + 1}y{j + 1}) ? p(x{i + 1})*p(y{j + 1}) => '
            if abs(xy[i][j] - x[i] * y[j]) > EPS:
                s += f'{xy[i][j]} != {x[i]} * {y[j]} = {round(x[i] * y[j], eps_sign)}\n'
            else:
                s += f'{xy[i][j]} = {x[i]} * {y[j]}\n'
    return s


def get_cond_prob(xy, prob, s1):
    xiyj, s = [], "\nУсловные вероятности\n"
    for i in range(len(xy)):
        new_mas = []
        for j in range(len(xy[0])):
            res = round(xy[i][j] / prob[j], eps_sign)
            s += f'p({s1[0]}{i + 1}|{s1[1]}{j + 1}) = p({s1[0]}{i + 1}{s1[1]}{j + 1})/p({s1[1]}{j + 1}) ' \
                 f'= {xy[i][j]} / {prob[j]} = {res}\n'
            new_mas.append(res)
        xiyj.append(new_mas)
    return xiyj, s


def get_cond_prob_xiyj(xy, y):
    return get_cond_prob(xy, y, "xy")


def get_cond_prob_yjxi(xy, x):
    return get_cond_prob(np.transpose(xy), x, "yx")


def get_prob_from_cond_prob(cond_prob, prob, s1):
    new_prob, s = [], f'\nВероятности событий ансамбля {s1[0]}:\n'

    for i in range(len(cond_prob)):
        yi = 0
        s += f'p({s1[0]}{i + 1}) = ' +\
             " + ".join([f'p({s1[1]}{j}) * p({s1[0]}{i + 1}|{s1[1]}{j})' for j in range(1, len(prob) + 1)]) +\
             " = " +\
             " + ".join([f'{prob[j]} * {cond_prob[i][j]}' for j in range(len(prob))])
        for j in range(len(prob)):
            yi += prob[j] * cond_prob[i][j]
        s += f' = {yi}\n'
        new_prob.append(round(yi, eps_sign))
    return new_prob, s


def get_prob_from_xiyj(cond_prob, prob):
    return get_prob_from_cond_prob(cond_prob, prob, "xy")


def get_prob_from_yjxi(cond_prob, prob):
    return get_prob_from_cond_prob(cond_prob, prob, "yx")


def get_xy_from_cond_prob(cond_prob, prob, s1):
    new_prob, s = [], "\nВероятность событий P(xiyj):\n"
    for i in range(len(cond_prob)):
        x = []
        for j in range(len(cond_prob[i])):
            res = round(cond_prob[i][j] * prob[j], eps_sign)
            s += f'p({s1[0]}{i + 1}{s1[1]}{j + 1}) = p({s1[0]}{i + 1}|{s1[1]}{j + 1}) * p({s1[1]}{j + 1}) = {res}\n'
            x.append(res)
        new_prob.append(x)
    return new_prob, s


def get_xy_from_xiyj(xiyj, y):
    return get_xy_from_cond_prob(xiyj, y, "xy")


def get_xy_from_yjxi(yjxi, x):
    xy, s = get_xy_from_cond_prob(yjxi, x, "yx")
    return np.transpose(xy), s


def get_entropy_formula(len_prob, s):
    return " + ".join(f'p({s}{i})*log(p({s}{i}))' for i in range(1, len_prob + 1))


def get_entropy_text(prob):
    return " + ".join([str(round(-1 * el * math.log2(el), eps_sign)) if el > 0 else "0" for el in prob])


def get_entropy_x(x):
    hx = get_entropy(x)
    s = "\nЭнтропия H(X):\nH(X) = -(" + get_entropy_formula(len(x), "x") + ") = " +\
        get_entropy_text(x) + f' = {hx}\n'
    return s


def get_entropy_y(y):
    hy = get_entropy(y)
    s = "\nЭнтропия H(Y):\nH(Y) = -(" + get_entropy_formula(len(y), "y") + ") = " +\
        get_entropy_text(y) + f' = {hy}\n'
    return s


def get_entropy(prob):
    h = 0
    for el in prob:
        if el > 0:
            h += round(el * math.log2(el), eps_sign)
    return round(-1 * h, eps_sign)


def get_entropy_xy(xy):
    h = 0
    s = "\nЭнтропия совместного ансамбля:\nH(XY) = " + \
        f'-СУММ(p(xiyj)*log(p(xiyj))), i=1..{len(xy)}, j=1..{len(xy[0])}) = ' + \
        " + ".join([f'({get_entropy_text(el)})' for el in xy])
    for el in xy:
        h += get_entropy(el)
    h = round(h, eps_sign)
    s += f' = {h}\n'
    return h, s


def get_total_cond_entropy(cond_prob, prob, s1):
    h = 0
    cond_prob = np.transpose(cond_prob)
    s = f'\nПолная условная энтропия:\nH{s1[0]}({s1[1]}) = ' + \
        f'СУММ(p({s1[0]}i)*H{s1[0]}i({s1[1]}), i=1..{len(prob)}) = ' + \
        " + ".join([f'{prob[i]} * ({get_entropy_text(cond_prob[i])})' for i in range(len(cond_prob))]) + " = " +\
        " + ".join([f'{prob[i]} * ({get_entropy(cond_prob[i])})' for i in range(len(cond_prob))])
    for i in range(len(cond_prob)):
        h += get_entropy(cond_prob[i]) * prob[i]
    h = round(h, eps_sign)
    s += f' = {h}\n'
    return h, s


def get_total_cond_entropy_hxy(yjxi, x):
    return get_total_cond_entropy(yjxi, x, "xy")


def get_total_cond_entropy_hyx(xiyj, y):
    return get_total_cond_entropy(xiyj, y, "yx")


def check(hxy, hx, hy, h_total_x, h_total_y):
    if abs(hxy - (hx + h_total_x)) > EPS or abs(hxy - (hy + h_total_y)) > EPS:
        return False
    return True


def get_text(prob, s1):
    s = ""
    for i in range(len(prob)):
        s += f'p({s1}{i + 1}) = {prob[i]}\n'
    return s


def get_text2(prob, s1, s2):
    s = ""
    for i in range(len(prob)):
        for j in range(len(prob[i])):
            s += f'p({s1}{i + 1}{s2}{j + 1}) = {prob[i][j]}\n'
    return s


def do_ex_1(xy):
    s = "Дано:\n" + get_text2(xy, "x", "y") +\
        "\nНайти: p(xi), p(yj), p(xi|yj), p(yj|xi), являются ли ансамбли независимыми?\n"

    x, s1 = get_x_from_xy(xy)
    y, s2 = get_y_from_xy(xy)
    cond_prob_xiyj, s3 = get_cond_prob_xiyj(xy, y)
    cond_prob_yjxi, s4 = get_cond_prob_yjxi(xy, x)
    s += s1 + s2 + is_independent_text(xy, x, y) + s3 + s4
    return s


def do_ex_2(xy):
    s = "Дано:\n" + get_text2(xy, "x", "y") + "\nНайти: H(X), H(Y), H(XY), Hy(X), Hx(Y)\n"

    x, s1 = get_x_from_xy(xy)
    y, s2 = get_y_from_xy(xy)
    hx, hy = get_entropy(x), get_entropy(y)
    hxy, s3 = get_entropy_xy(xy)
    yjxi, s4 = get_cond_prob_yjxi(xy, x)
    hx_y, s4 = get_total_cond_entropy_hxy(yjxi, x)
    xiyj, s5 = get_cond_prob_xiyj(xy, y)
    hy_x, s6 = get_total_cond_entropy_hyx(xiyj, y)
    s += s1 + s2 + get_entropy_x(x) + get_entropy_y(y) + s3 + s4 + s5 + s6
    return s


def do_example_2(x, yjxi):
    s = "Дано:\n" + get_text(x, "x") + "\n" + get_text2(yjxi, "y", "|x") + \
        "\nНайти: H(X), H(Y), H(XY), Hy(X), Hx(Y)\n"

    y, s1 = get_prob_from_yjxi(yjxi, x)
    hx, hy = get_entropy(x), get_entropy_y(y)
    xy, s2 = get_xy_from_yjxi(yjxi, x)
    hxy, s3 = get_entropy_xy(xy)
    hx_y, s4 = get_total_cond_entropy_hxy(yjxi, x)
    xiyj, s5 = get_cond_prob_xiyj(xy, y)
    hy_x, s6 = get_total_cond_entropy_hyx(xiyj, y)
    s += get_entropy_x(x) + get_entropy_y(y) + s1 + s2 + s3 + s4 + s5 + s6
    return s


def write_ex(file_name, s):
    file = 'output_' + file_name + '.txt'
    with open(file, 'w') as f:
        f.write(s)
        f.write("\n")
    f.close()


if __name__ == '__main__':
    xy1 = np.array([[0.038, 0.025, 0.067, 0.051], [0.061, 0.071, 0.107, 0.088], [0.072, 0.072, 0.054, 0.294]])
    str1 = do_ex_1(xy1)
    str2 = do_ex_2(xy1)
    write_ex("test1_", str1)
    write_ex("test2_", str2)
