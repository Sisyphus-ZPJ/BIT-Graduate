x = [5,9,8,4,7] # 五轮中硬币为正面的次数

pA = 0.6 # A为正面的概率
pB = 0.5 # B为正面的概率

T1 = [0,0,0,0,0]
T2 = [0,0,0,0,0]
T3 = [0,0,0,0,0]
T4 = [0,0,0,0,0]

if __name__ == '__main__':
    iterations = 10
    for iters in range(iterations):
        print('iteration '+str(iters)+': pA = '+str(pA)+' pB = '+str(pB))
        # E-step
        for i in range(len(x)):
            t1 = (pA)**x[i] * (1-pA)**(10-x[i]) / ((pA)**x[i] * (1-pA)**(10-x[i]) + (pB)**x[i]*(1-pB)**(10-x[i]))
            t2 = 1 - t1

            T1[i] = x[i] * t1       # A 正面
            T2[i] = (10-x[i]) * t1  # A 反面
            T3[i] = x[i] * t2       # B 正面
            T4[i] = (10-x[i]) * t2  # B 反面

        # M-step
        pA = sum(T1)/(sum(T1) + sum(T2))
        pB = sum(T3)/(sum(T3) + sum(T4))

        iterations -= 1

    print('Final: pA = '+str(pA)+' pB = '+str(pB))
    