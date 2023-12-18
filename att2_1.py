"""Записывать p(xi|yj) в виде
[[p(a|a), p(a|b), p(a|c)], [p(b|a), p(b|b), p(b|c)], [p(c|a), p(c|b), p(c|c)]]"""

import math
import numpy as np

EPS = 1e-6
eps_sign = 7
ABC = "abc"


def get_prob_relation(prob):
    s = "Связь вероятностей состояний марковской цепи на i-м и (i + 1)-м шагах:\n"
    s += "p(Xi+1 = sk) = СУММ(p(Xi=sj) * p(Xi+1 = sk | Xi = sj), при sj={a,b,c}), где sk={a,b,c}\n\n"
    for i in range(len(prob)):
        s += f'P{ABC[i]} = '
        s += " + ".join([f'{prob[i][j]} * P{ABC[j]}' if prob[i][j] != 0 else "0" for j in range(len(prob[i]))])
        s += '\n'
    return s + "\n"


def get_prob_system(prob, s, res):
    for i in range(len(prob)):
        s += " + ".join([f'{round(prob[i][j], eps_sign)} * P{ABC[j]}' if prob[i][j] != 0 else "0" for j in range(len(prob[i]))])
        s += f' = {res[i]}'
        s += '\n'
    s += " + ".join([f'P{ABC[i]}' for i in range(len(ABC))]) + ' = 1\n\n'
    return s


def get_prob_system1(prob):
    prob_ = prob.copy()
    for i in range(len(prob_)):
        prob_[i][i] -= 1
    s = "Система уравнений, дополненная ограничением СУММ(Psj = 1, при sj={a,b,c}):\n"
    return prob_, get_prob_system(prob_, s, "000")


def get_prob_system2(prob):
    prob_ = prob.copy()
    for i in range(len(prob_)):
        prob_[2][i] = 1
    return prob_


def get_system_coeff(prob):
    prob_ = prob.copy()
    for i in range(len(prob_)):
        prob_[i][i] -= 1
    return prob_


def get_system_coeff2(prob):
    prob_ = prob.copy()
    for i in range(len(prob_)):
        prob_[2][i] = 1
    return prob_


def get_solution(matrix):
    return np.linalg.solve(matrix, np.array([0., 0., 1.]))


def get_solution_text(system):
    prob = get_solution(system)
    s = "Вероятности стационарного распределения заданной марковской цепи:\n"
    for i in range(len(prob)):
        s += f'P{ABC[i]} = {round(prob[i], eps_sign)}\n'
    return prob, s + '\n '


def get_joint_prob(cond, arr_):
    x = []
    s = "Совместные вероятности p(Xi=sj, Xi+1=sk)\n"
    for i in range(len(cond)):
        el = cond[i]
        for j in range(len(el)):
            res = round(arr_[j] * el[j], eps_sign)
            s += f'p(Xi={ABC[j]}, Xi+1={ABC[i]}) = P{ABC[j]}*P({ABC[j]}|{ABC[i]}) = ' \
                 f'{round(arr_[j], eps_sign)} * {round(el[j])} = {res}\n'
            x.append(res)
    return x, s + "\n"


def get_entropy_text(prob):
    s = "-(" + " + ".join([f'{round(el, eps_sign)}log2({round(el, eps_sign)})' if el > 0 else "0" for el in prob]) +\
        ") =\n= -(" + \
        " + ".join([str(round(-1 * el * math.log2(el), eps_sign)) if el > 0 else "0" for el in prob]) + ")"
    return s


def get_entropy_xi(x):
    hx = get_entropy(x)
    s = "Энтропия H(Xi):\nH(Xi) = -СУММ(p(Xi = sj)*log2(p(Xi = sj))),при Sj={a,b,c} " + \
        "= -СУММ(Psj*log2(Psj)), Sj={a,b,c} =\n" + \
        "= " + get_entropy_text(x) + f' = {hx}\n\n'
    return hx, s


def get_entropy_xixi1(x):
    hxx = get_entropy(x)
    s = "Энтропия H(XiXi+1):\n" + get_entropy_text(x) + f'= {hxx}\n'
    return hxx, s


def get_cond_entropy1(prob, cond_prob):
    h = 0
    s = f'\nУсловная энтропия Hxi(Xi+1)\n\n' + "Вариант 1\n" + \
        'Hxi(Xi+1) = СУММ(Psj*СУММ(P(sk|sj) * log2(P(sk|sj)), sk={a,b,c}), sj={a,b,c}) = \n= ' + \
        " +\n+ ".join([f'P{i} * [{" + ".join([f"P({i}|{j})*log2(P({i}|{j}))" for j in ABC])}]' for i in ABC]) + " =\n= " + \
        " + ".join([f'{round(prob[i], eps_sign)} * ({get_entropy(cond_prob[i])})' for i in range(len(cond_prob))])
    for i in range(len(cond_prob)):
        h += get_entropy(cond_prob[i]) * prob[i]
    h = round(h, eps_sign)
    s += f' = {h}\n\n'
    return h, s


def get_cond_entropy2(entr1, entr2):
    h = round(entr1 - entr2, eps_sign)
    s = "Вариант 2\n" + f"H(XiXi+1) - H(Xi) = {entr1} - {entr2} = {h}\n"
    return h, s


def get_entropy(prob):
    h = 0
    for el in prob:
        if el > 0:
            h += round(el * math.log2(el), eps_sign)
    return round(-1 * h, eps_sign)


def get_text(prob):
    s = ""
    for i in range(len(prob)):
        for j in range(len(prob[i])):
            s += f'P({ABC[i]}|{ABC[j]}) = {prob[i][j]}\n'
    return s


def get_res(xy):
    s = "Дано:\n" + get_text(xy) + "\nНайти: Xi, H(Xi), H(XiXi+1), HXi(Xi+1)\n\n"
    s += get_prob_relation(xy)
    system1, s1 = get_prob_system1(xy)
    system2 = get_prob_system2(system1)
    arr_, s3 = get_solution_text(system2)
    hx, s4 = get_entropy_xi(arr_)
    p, s5 = get_joint_prob(xy, arr_)
    hxx, s6 = get_entropy_xixi1(p)
    h1, s7 = get_cond_entropy1(arr_, xy)
    h2, s8 = get_cond_entropy2(hxx, hx)
    return s + s1 + s3 + s4 + s5 + s6 + s7 + s8


def write_ex(file_name, s):
    file = 'output_' + file_name + '.txt'
    with open(file, 'w') as f:
        f.write(s)
        f.write("\n")
    f.close()


if __name__ == '__main__':
    xy1 = np.array([[0.33, 0.08, 0.26], [0.31, 0.67, 0.67], [0.36, 0.25, 0.07]])
    write_ex("att_ex1", get_res(xy1))
