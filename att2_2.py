import math

EPS = 1e-6
eps_sign = 7


def get_haffman(prob, dict_01):
    last_probs = (prob[-2][0] + prob[-1][0], round(prob[-2][1] + prob[-1][1], eps_sign))
    new_prob = prob[:-2]
    new_prob += (last_probs,)
    dict_ = check_01(dict_01, prob[-2], prob[-1])
    return dict_, get_sort(new_prob)


def get_sort(prob):
    return sorted(prob, key=lambda x: x[1], reverse=True)


def get_text(prob):
    return ", ".join([f'{tuple_[0]} : {tuple_[1]}' for tuple_ in prob]) + "\n"


def get_text_dict(dict_):
    return ", ".join([f'{k}: ({v})' for k, v in dict_.items()]) + "\n"


def set_01(dict_01, z1, z2, s):
    for el in z1:
        dict_01[el] = s[0] + dict_01[el]
    for el in z2:
        dict_01[el] = s[1] + dict_01[el]
    return dict_01


def check_01(dict_01, prob1, prob2):
    z1, z2 = prob1[0], prob2[0]
    p_z1, p_z2 = prob1[1], prob2[1]
    sum_z1, sum_z2 = sum(z1), sum(z2)
    if len(z1) < len(z2) or (len(z1) == len(z2) and p_z1 > p_z2) or \
            (len(z1) == len(z2) and abs(p_z1 - p_z2) < EPS and (sum_z2 - sum_z1 > EPS)):
        dict_01 = set_01(dict_01, z1, z2, "01")
    else:
        dict_01 = set_01(dict_01, z1, z2, "10")
    return dict_01


def get_l(dict_01, my_dict):
    return round(sum([len(dict_01[i + 1]) * my_dict[i][1] for i in range(len(my_dict))]), eps_sign)


def get_l_str(dict_01, my_dict):
    l_ = get_l(dict_01, my_dict)
    s = f'L = СУММ(li * pi), i=1..n =\n' + '= ' + \
        " + ".join([f'{len(dict_01[i + 1])} * {my_dict[i][1]}' for i in range(len(my_dict))]) + f' =\n= {l_}\n\n'
    return l_, s


def get_entropy(prob):
    h = 0
    for el in prob:
        if el[1] > 0:
            h += round(el[1] * math.log2(el[1]), eps_sign)
    return round(-1 * h, eps_sign)


def get_entropy_text(prob):
    h = get_entropy(prob)
    s = "H = СУММ(pi * log2(pi)), i=1..n =\n= -(" + \
        " + ".join([f'{el[1]}*log2({el[1]})' if el[1] > 0 else "0" for el in prob]) + ") =\n= -(" + \
        " + ".join(
            [str(round(-1 * el[1] * math.log2(el[1]), eps_sign)) if el[1] > 0 else "0" for el in prob]) + ") =\n" + \
        f"=  {str(h)}\n\n"
    return h, s


def ger_r(l_, h):
    return round(l_ - h, eps_sign)


def get_r_text(l_, h):
    r = ger_r(l_, h)
    return r, f"L - H = {r}\n\n"


def get_k(dict_01):
    return round(sum([math.pow(2, -len(value)) for value in dict_01.values()]), eps_sign)


def get_k_str(dict_01):
    s = "Неравенство Крафта: k = СУММ(2^(-li), при i=1..n) <= 1\n"
    k = get_k(dict_01)
    s += f'K = СУММ(2^li), i=1..n =\n' + '= ' + \
        " + ".join([f'{math.pow(2, -len(value))}' for value in dict_01.values()]) + f' =\n= {k}\n\n'
    return k, s


def get_haf_res(my_dict):
    s = "Дано:\n" + get_text(my_dict) + "\nНайти код Хаффмана:\n\n"
    new_prob = get_sort(my_dict).copy()
    dict_01 = {}
    for el in new_prob:
        dict_01[el[0][0]] = ""

    s += get_text(new_prob)
    s += get_text_dict(dict_01) + "\n"
    while len(new_prob) > 1:
        dict_01, new_prob = get_haffman(new_prob, dict_01)
        s += get_text(new_prob)
        s += get_text_dict(dict_01) + "\n"

    l_, s2 = get_l_str(dict_01, my_dict)
    h, s3 = get_entropy_text(my_dict)
    r, s4 = get_r_text(l_, h)
    return s + s2 + s3 + s4


def check(el_mid_next, el0, mid):
    if (mid - el0) > (el0 + el_mid_next - mid):
        return 1
    return 0


def get_shen_fano(prob, prob_2, dict_01):
    el_mid = round(sum([el[1] for el in prob]) / 2.0, eps_sign)
    el0, el1, i = 0, 0, 0
    prob0 = []
    prob1 = []
    while check(prob_2[i][1], el0, el_mid):
        el0 += prob_2[i][1]
        prob0.append(prob_2[i][0][0])
        i += 1
    while i < len(prob_2):
        prob1.append(prob_2[i][0][0])
        el1 += prob_2[i][1]
        i += 1
    new_prob = [(prob0, round(el0, eps_sign)), (prob1, round(el1, eps_sign))]
    dict_ = check_01_shen(dict_01, new_prob[-2], new_prob[-1])
    return dict_, new_prob


def set_01_shen(dict_01, z1, z2, s):
    for el in z1:
        dict_01[el] = dict_01[el] + s[0]
    for el in z2:
        dict_01[el] = dict_01[el] + s[1]
    return dict_01


def check_01_shen(dict_01, prob1, prob2):
    z1, z2 = prob1[0], prob2[0]
    p_z1, p_z2 = prob1[1], prob2[1]
    sum_z1, sum_z2 = sum(z1), sum(z2)
    if len(z1) < len(z2) or (len(z1) == len(z2) and p_z1 > p_z2) or \
            (len(z1) == len(z2) and abs(p_z1 - p_z2) < EPS and (sum_z2 - sum_z1 > EPS)):
        dict_01 = set_01_shen(dict_01, z1, z2, "01")
    else:
        dict_01 = set_01_shen(dict_01, z1, z2, "10")
    return dict_01


def get1(prob, my_dict, dict_01):
    s = ""
    len_prev = 0
    for el in prob:
        if len(el[0]) != 1:
            dict_01, prob = get_shen_fano([el], my_dict[len_prev: len_prev + len(el[0])], dict_01)
            s += get_text(prob) + get_text_dict(dict_01) + "\n"
            s += get1(prob, my_dict[len_prev: len_prev + len(el[0])], dict_01)
        len_prev += len(el[0])
    return s


def get_shen_fano_res(my_dict):
    s = "Дано:\n" + get_text(my_dict) + "\nНайти код Шеннона-Фано:\n\n"
    new_prob = get_sort(my_dict).copy()
    s += get_text(new_prob) + "\n"
    dict_01 = {}
    for el in new_prob:
        dict_01[el[0][0]] = ""
    new_prob = [([el[0][0] for el in new_prob], sum([new_prob[i][1] for i in range(len(new_prob))]))]

    s += get_text(new_prob)
    s += get_text_dict(dict_01) + "\n"
    s += get1(new_prob, get_sort(my_dict).copy(), dict_01)

    l_, s2 = get_l_str(dict_01, my_dict)
    h, s3 = get_entropy_text(my_dict)
    r, s4 = get_r_text(l_, h)
    k, s5 = get_k_str(dict_01)
    return s + s2 + s3 + s4 + s5


def write_ex(file_name, s):
    file = 'output_' + file_name + '.txt'
    with open(file, 'w') as f:
        f.write(s)
        f.write("\n")
    f.close()


if __name__ == '__main__':
    dict1 = [([1], 0.381), ([2], 0.09), ([3], 0.085), ([4], 0.07), ([5], 0.19),
             ([6], 0.066), ([7], 0.01), ([8], 0.018), ([9], 0.026), ([10], 0.064)]
    write_ex("atta_haffman_ex1", get_haf_res(dict1))
    write_ex("atta_shen_fano_ex1", get_shen_fano_res(dict1))
