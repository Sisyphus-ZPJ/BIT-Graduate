state_graph = [[],
               [0.4, 0.1, 0.2, 0.3],
               [0.4, 0.1, 0.1, 0.4],
               [0.2, 0.3, 0.3, 0.2],
               [0.1, 0.4, 0.4, 0.1],
               []]

state_trans = [[[0.5, 1], [0.5, 2]],
               [[0.2, 1], [0.8, 3]],
               [[0.8, 2], [0.2, 4]],
               [[0.4, 3], [0.6, 5]],
               [[0.1, 4], [0.9, 5]]]

def update(state_now, target, state_history, p_now):
    P = 0
    # Start
    if state_now == 0:
        for trans in state_trans[state_now]:
            P += trans[0] * update(state_now=trans[1], target = target, state_history = state_history, p_now=p_now*trans[0])
    # Other Node
    else:
        # End
        if len(target) == 0:
            # Correct
            if state_now == 5:
                print('Sequence:', state_history,  ' P='+str(p_now))
                return 1
            # Error
            else:
                return 0
        # Midlle
        else:
            # Error
            if state_now == 5:
                return 0

            # Calculate propability
            p_generate = state_graph[state_now][int(target[0])]
            for trans in state_trans[state_now]:
                state_history_now = state_history + str(state_now)
                P += p_generate * trans[0] * update(state_now=trans[1], target = target[1:], state_history = state_history_now, p_now=p_now * p_generate * trans[0])

    return P


if __name__ == '__main__':
    string = 'TAGA'
    generates = []
    
    for char in string:
        if char == 'A':
            generates.append(0)
        elif char == 'C':
            generates.append(1)
        elif char == 'G':
            generates.append(2)
        elif char == 'T':
            generates.append(3)
        else:
            raise ValueError('chars of string must be ACGT')

    P = update(state_now = 0, target = generates, state_history = '', p_now = 1)

    print('Total propability:', P)
