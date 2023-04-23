#for calculating derivatives
#input: math equation in string form
#output: derivative in string form


def arithmetic_calc(a, b, operator):
    if operator == '+':
        return a + b
    elif operator == '-':
        return a - b
    elif operator == '*':
        return a * b
    elif operator == '/':
        return a / b


def parentheses(p):
    ops_list = ['+', '-', '*', '/']

    #remove parentheses
    ret = p[1:-1]

    op_index = 0
    #separate values
    for op in ops_list:
        op_index = ret.find(op)
        if op_index != -1:
            break

    front = int(ret[:op_index])
    back = int(ret[op_index + 1:])

    return front, back, ret[op_index]


def find_deriv(equation):
    x_location = equation.find('x')
    if x_location == 0:
        coef = 1
    else:
        coef = int(equation[:x_location])

    symbol_location = equation.find('^')

    # for power with parenthesis
    if equation[symbol_location+1] == '(':

        #make sure to include the parenthesis
        paren = equation[symbol_location+1:]

        #make it suitable for arithmetic_calc function
        a, b, c = parentheses(paren)
        power = arithmetic_calc(a, b, c)

    else:
        power = int(equation[symbol_location + 1:])

    variable = equation[x_location]
    derivative = str(coef * power) + variable + '^' + str(power-1)

    return derivative


z = 'x^(5/2)'
print(find_deriv(z))
