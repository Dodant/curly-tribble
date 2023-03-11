def solution(bin1, bin2):
    bin1 = '0b' + bin1
    bin2 = '0b' + bin2
    return str(bin(int(str(int(bin1, 2)+int(bin2, 2)))))[2:]