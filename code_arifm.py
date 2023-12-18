import math
eps = 15


def get_q(p, n):
    s = "Кумулятивные вероятности:\n" \
        "q(Si) = q(Si-1) + p(Si-1)\n"
    q = [0]
    p1 = [0] + p
    for i in range(1, n + 1):
        q.append(round(q[i - 1] + p1[i - 1], eps))
        s += f'q(s{i}) = {q[i - 1]} + {p1[i - 1]} = {q[i]}\n'
    # s += ",\n".join(f'q(s{i}) = {q[i]}' for i in range(1, n + 1)) + "\n\n"
    s += "\n"
    return s, q[1:]


def get_g(p, sik):
    s = "Длина интервала вероятностей, соответствующая последовательности Sik:\n" \
        "G(Sik) = p(Si)*G(Sik-1)\n" \
        "G(-) = 1\n"
    n = len(sik)
    p1 = [0] + p
    g = [1]
    for i in range(1, n + 1):
        g.append(round(p1[sik[i - 1]] * g[i - 1], eps))
        s += "G(s" + "s".join(str(sik[j]) for j in range(i)) + f") = {p1[sik[i - 1]]} * {g[i - 1]} = {g[-1]}\n"
    s += "\n"
    return s, g


def get_f(q, g, sik):
    s = "Нижняя граница вероятностей, соответствующая последовательности Sik:\n" \
        "F(Sik) = F(Sik-1) + q(Si)*G(Sik-1)\n" \
        "F(-) = 0\n"
    f = [0]
    for i in range(1, len(sik) + 1):
        f.append(round(f[i - 1] + q[sik[i - 1] - 1] * g[i - 1], eps))
        s += "F(s" + "s".join(str(sik[j]) for j in range(i)) + f") =" \
                                                               f" {f[i - 1]} + {q[sik[i - 1] - 1]} * {g[i - 1]} = " \
                                                               f"{f[-1]}\n"
    s += "\n"
    return s, f


def get_l(g):
    l_ = math.ceil(-math.log2(g[-1])) + 1
    s = f"Длина кодового слова: (округление вверх)\n" \
        f"L = [-log2(G(s))] + 1 = [-log2({g[-1]})] + 1 =  {l_}\n\n"
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


def get_x(f, g, l_):
    x = f + g / 2
    x = get_bin(x, l_)
    s = f"Искомое кодовое слово:\n" \
        f"x = bin(F(s) + G(s)/2) = bin({f} + {g} / 2) = bin({f + g / 2}) = {x}\n\n"
    return s, x


def get_decode_x(x1):
    decimal_number = 0
    power = -1
    i = x1.find(".")
    x = x1[i + 1:]
    for digit in x:
        digit = int(digit)
        decimal_number += round(digit * (2 ** power), eps)
        power -= 1

    s = f'Декодированный код:\n' \
        f'x = {x} = {x1} =\n= '
    s += " + ".join(f'{x[i]} * 2^(-{i + 1})' for i in range(len(x))) + " =\n= "
    s += " + ".join(f'{round(int(x[i]) * (2 ** (-(i + 1))), eps)}' for i in range(len(x))) + " =\n= "
    s += f'{decimal_number}\n\n'
    return s, decimal_number


def get_decode(f, g, q, sik, x, n):
    s = "Декодирование:\n"
    s += "      Проверка  F         G         q         F + q * G\n\n"
    s2 = ""
    for i in range(len(sik)):
        check_res = 1
        s += f"{i + 1}\n"
        for j in range(n):
            res = round(f[i] + q[j] * g[i], eps)
            if res < x:
                check = "ИСТИНА"
                if j == n - 1:
                    s2 += f's{j + 1}'
            else:
                check = "ЛОЖЬ"
                if check_res:
                    s2 += f's{j}'
                check_res = 0
            s += f"S{j + 1}    {check}     {f[i]}       {g[i]}          {q[j]}        {res}\n"
        s += "\n"
    s += f'Декодированное сообщение: {s2}\n'
    return s


def get_dano_text(n, prob, sik):
    s = "Алфавит S = {" + ', '.join(f's{i}' for i in range(1, n + 1)) + "}\n"
    s += "Распределение вероятностей: " + ', '.join(f'p(s{i + 1}) = {prob[i]}' for i in range(n)) + "\n"
    s += "Последовательность s = (" + ''.join(f's{el}' for el in sik) + ")\n"
    return s


def get_res(n, prob, sik):
    s = "Дано:\n" + get_dano_text(n, prob, sik) + "\nПроизвести арифметическое кодирование и декодирование\n\n"
    s1, q = get_q(prob, n)
    s2, g = get_g(prob, sik)
    s3, f = get_f(q, g, sik)
    s4, l_ = get_l(g)
    s5, x = get_x(f[-1], g[-1], l_)
    s6, x2 = get_decode_x(x)
    s7 = get_decode(f, g, q, sik, x2, len(prob))
    return s + s1 + s2 + s3 + s4 + s5 + s6 + s7


def write_ex(file_name, s):
    file = 'output_' + file_name + '.txt'
    with open(file, 'w') as f:
        f.write(s)
        f.write("\n")
    f.close()


if __name__ == '__main__':
    s_n = 3
    prob1 = [0.1, 0.6, 0.3]
    sik1 = [2, 3, 2, 1, 2]
    write_ex("arf1", get_res(s_n, prob1, sik1))
