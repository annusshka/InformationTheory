import math
eps = 7


def get_theta(p, q):
    theta = []
    s = "Найти theta:\n"
    for i in range(len(p)):
        theta.append(round(q[i] + p[i] / 2, eps))
        s += f'theta{i + 1} = q{i + 1} + p{i + 1} / 2 = {q[i]} + {p[i]} / 2 = {theta[-1]}\n'
    s += "\n"
    return s, theta


def get_q(p):
    q = [0]
    s = "Найти q:\n" + 'q1 = 0\n'
    for i in range(1, len(p)):
        q.append(round(q[i - 1] + p[i - 1], eps))
        s += f'q{i + 1} = q{i} + p{i} = {q[i - 1]} + {p[i - 1]} = {q[-1]}\n'
    s += "\n"
    return s, q


def get_l(p):
    l_, log2 = [], []
    s = f"Длина кодового слова: (округление вверх)\n"
    for i in range(len(p)):
        log2.append(-math.log2(p[i]))
        l_.append(math.ceil(log2[i]) + 1)
        s += f"L{i + 1} = [-log2(p)] + 1 = [-log2({p[i]})] + 1 = [{round(log2[i], eps)}] + 1 = {l_[i]}\n"
    s += "\n"
    return s, l_


def float_to_binary(x, n):
    x_scaled = round(x * 2 ** n)
    x = '{:0{}b}'.format(x_scaled, 1 + n)
    return str(x[0]) + "." + str(x[1:])


def get_bin(x, l_):
    dec, whole = math.modf(x)
    whole = int(whole)
    res = bin(whole).lstrip("0b") + "."

    i = 0
    while i < l_:
        dec *= 2
        dec, whole = math.modf(dec)
        res += str(int(whole))
        i += 1

    return res


def get_theta_01_default(theta):
    theta_01 = []
    s = "theta в двоичной системе:\n"
    for i in range(len(theta)):
        theta_01.append(get_bin(theta[i], 10))
        s += f"x{i + 1} = {theta_01[i]}\n"
    s += "\n"
    return s, theta_01


def get_theta_01(theta, l_):
    theta_01 = []
    s = "theta в двоичной системе:\n"
    for i in range(len(theta)):
        theta_01.append(get_bin(theta[i], l_[i]))
        s += f"x{i + 1} = {theta_01[i]}\n"
    s += "\n"
    return s, theta_01


def get_mid_l(p, l_1):
    l_ = round(sum([p[i] * l_1[i] for i in range(len(l_1))]), eps)
    s = "Средняя длина:\n" + f'L = СУММ(li * pi), i=1..n =\n' + '= ' + \
        " + ".join([f'{p[i]} * {l_1[i]}' for i in range(len(l_1))]) + f' =\n= {l_}\n\n'
    return s, l_


def get_entropy(prob):
    h = 0
    for el in prob:
        if el > 0:
            h += round(el * math.log2(el), eps)
    return round(-1 * h, eps)


def get_entropy_text(prob):
    h = get_entropy(prob)
    s = "Энтропия:\n" + "H = -СУММ(pi * log2(pi)), i=1..n =\n= -(" + \
        " + ".join([f'{el}*log2({el})' if el > 0 else "0" for el in prob]) + ") =\n= -(" + \
        " + ".join(
            [str(round(-1 * el * math.log2(el), eps)) if el > 0 else "0" for el in prob]) + ") =\n" + \
        f"=  {str(h)}\n\n"
    return s, h


def get_r_text(l_, h):
    r = round(l_ - h, eps)
    s = "Избыточность:\n" + f"L - H = {r}\n\n"
    return s, r


def get_k(l_):
    s = "Неравенство Крафта: k = СУММ(2^(-li), при i=1..n) <= 1\n"
    k = round(sum([math.pow(2, -el) for el in l_]), eps)
    s += f'K = СУММ(2^-li), i=1..n =\n' + '= ' + \
         " + ".join([f'2^{-el}' for el in l_]) + " =\n= " + \
         " + ".join([f'{math.pow(2, -el)}' for el in l_]) + f' =\n= {k}\n\n'
    return s, k


def get_res(p):
    s = "Дан алфавит с вероятностями:\n" + '\n'.join(f'z{i + 1} = {p[i]}' for i in range(len(p))) + "\n\n" + \
        "Построить код Гилберта-Мура\n\n"
    s1, q = get_q(p)
    s2, theta = get_theta(p, q)
    s4, l_ = get_l(p)
    s5_1, x_def = get_theta_01_default(theta)
    s5, x = get_theta_01(theta, l_)
    s6, mid_l = get_mid_l(p, l_)
    s7, h = get_entropy_text(p)
    s8, r = get_r_text(mid_l, h)
    s9, k = get_k(l_)
    return s + s1 + s2 + s4 + s5_1 + s5 + s6 + s7 + s8 + s9


def write_ex(file_name, s):
    file = 'output_' + file_name + '.txt'
    with open(file, 'w') as f:
        f.write(s)
        f.write("\n")
    f.close()


if __name__ == '__main__':
    p1 = [0.274, 0.09, 0.1, 0.115, 0.19, 0.09, 0.034, 0.022, 0.025, 0.06]
    write_ex("code1", get_res(p1))
