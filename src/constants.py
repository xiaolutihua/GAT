K = 1e3
M = 1e6
G = 1e9
T = 1e12


def hstr(num: float) -> str:
    """输入一个浮点数，返回人类友好浮点数字符串表达形式。

    - 8 -> '8'
    - 1600 -> '1.60K'
    - 3200000000 -> '3.20G'
    """

    if num >= T:
        return f'{num/T:.2f}T'
    elif num >= G:
        return f'{num/G:.2f}G'
    elif num >= M:
        return f'{num/M:.2f}M'
    elif num >= K:
        return f'{num/K:.2f}K'
    return str(num)
