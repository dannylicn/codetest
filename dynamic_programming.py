# !usr/bin/python3
"""
Different graph algorithms implementation
Arash Tehrani
"""
class DynamicProgramming(object):
    def __init__(self):
        pass
    
    def longest_increasing_subsequence(self, arr):
        """
        longest increasing subsequence
        inputs:
            - arr: list, input array
        outputs:
            - length of the maximum increasing subarray
            - the maximum increasing subarray
        """
        len_arr = [1]*len(arr)
        select_arr = list(range(len(arr)))
        
        for j in range(1, len(arr)):
            for i in range(j):
                if arr[i] < arr[j]:
                    if len_arr[j] < len_arr[i] + 1:
                        len_arr[j] = len_arr[i] + 1
                        select_arr[j] = i
                
        max_tuple = max([v for v in enumerate(len_arr)], key=lambda x:x[1])
        idx = max_tuple[0]
        n = max_tuple[1]
        idx_arr = [idx]
        
        while n > 1:
            idx_arr.append(select_arr[idx])
            idx = idx_arr[-1]
            n -= 1

        return max_tuple[1], [arr[i] for i in idx_arr][::-1]

    def knapsack01(self, w, val, W):
        """
        0/1 Knapsack problem
        inputs:
            - w: list of ints, weights of the items
            - val: list, value of the items
            - W: int, max allowable weight (budget)
        outputs:
            - max_val: max possible value
            - items: list of tuples, items picked
        """
        assert isinstance(W, int), 'W must be an integer'
        arr = [(i, j) for i, j in zip(val, w)]
        arr = sorted(arr, key=lambda x: x[1])

        score_mat = [[0]*(W+1) for i in range(len(arr))]
        for j in range(W+1):
            if j >= arr[0][1]:
                score_mat[0][j] = arr[0][0]
        
        for i in range(1, len(arr)):
            for j in range(1, W+1):
                if j < arr[i][1]:
                    score_mat[i][j] = score_mat[i-1][j]
                else:
                    score_mat[i][j] = max(score_mat[i-1][j], 
                                        arr[i][0] + score_mat[i-1][j - arr[i][1]]
                                        )
        # pick the elements based on the scores
        v = max_val = score_mat[-1][-1]
        idx = len(arr) - 1
        col = W
        items = []

        while v != 0:
            if score_mat[idx][col] == score_mat[idx-1][col]:
                idx -= 1
            else:
                col -= arr[idx][1]
                items.append(arr[idx])
                idx -= 1
            v = score_mat[idx][col]

        return max_val, items

    def unbounded_knapsack(self, w, val, W):
        """
        Unbounded Knapsack problem which allows repetition of items
        inputs:
            - w: list of ints, weights of the items
            - val: list, value of the items
            - W: int, max allowable weight (budget)
        outputs:
            - max_val: max possible value
            - items: list of tuples, items picked
        """
        assert isinstance(W, int), 'W must be an integer'
        arr = [(i, j) for i, j in zip(val, w)]
        arr = sorted(arr, key=lambda x: x[1])
        print(arr)
        score_mat = [[0]*(W+1) for i in range(len(arr))]
        for j in range(W+1):
            if j >= arr[0][1]:
                score_mat[0][j] = arr[0][0] + score_mat[0][j - arr[0][1]]
        
        for i in range(1, len(arr)):
            for j in range(1, W+1):
                if j < arr[i][1]:
                    score_mat[i][j] = score_mat[i-1][j]
                else:
                    score_mat[i][j] = max(score_mat[i-1][j], 
                                        arr[i][0] + score_mat[i][j - arr[i][1]]
                                        )
        # pick the elements based on the scores
        v = max_val = score_mat[-1][-1]
        idx = len(arr) - 1
        col = W
        items = []
        while v != 0:
            if idx == 0:
                col -= arr[idx][1]
                items.append(arr[idx])
            else:
                if score_mat[idx][col] == score_mat[idx-1][col]:
                    idx -= 1
                else:
                    col -= arr[idx][1]
                    items.append(arr[idx])
            v = score_mat[idx][col]

        return max_val, items

    def egg_dropping(self, n, k):
        """
        egg dropping problem
        * O(n*k^2) complexity
        inputs:
            - n: number of floors
            - k: number of eggs
        return:
            - int, numbner of attempts
        """
        T = [[0]*(n+1) for _ in range(k)]
        for j in range(n+1):
            T[0][j] = j
        
        for i in range(1, k):
            for j in range(1, n+1):
                if i > j:
                    T[i][j] = T[i-1][j]
                else:
                    sol = []
                    for jj in range(1, j+1):
                        sol.append(max(T[i-1][jj-1], T[i][j-jj]))
                    print(i, j, sol)
                    T[i][j] = 1 + min(sol)
        
        return T[-1][-1]

    def matrix_chain_multiplication(self, M_list):
        """
        Matrix chain multiplication
        inputs:
            M_list: list of lists of length 2, size of the amtrices
        outputs:
            int, minimum number of computation
            list, order of multiplications
        """
        n = len(M_list)
        T = [[0]*n for _ in M_list]
        order = [[0]*n for _ in M_list]
        
        for v in range(1, n):
            for i in range(n - v):
                j = i + v
                T[i][j] = float('inf')
                for k in range(i, j):
                    w = T[i][k] + T[k+1][j] + M_list[i][0]*M_list[k][1]*M_list[j][1]
                    if w < T[i][j]:
                        T[i][j] = w
                        order[i][j] = k
        
        def order_printer(i, j, order):
            if j - i == 1:
                return [[i],[j]]
            
            if order[i][j] == i:
                return [[i], order_printer(order[i][j]+1, j, order)]
            elif order[i][j]+1 == j: 
                return [order_printer(i, order[i][j], order), [j]]
            else:
                return [order_printer(i,order[i][j], order), 
                        order_printer(order[i][j]+1, j, order)]
        
        return T[0][-1], order_printer(0, n-1, order)

    def optimal_game_strategy(self, arr):
        """
        Consider a row of n coins of values v1 . . . vn, where n is even. We play 
        a game against an opponent by alternating turns. In each turn, a player selects
        either the first or last coin from the row, removes it from the row permanently, 
        and receives the value of the coin. Determine the maximum possible amount of 
        money we can definitely win if we move first.
        inputs:
            - arr: list, arr of value of the coins
        output:
            - scalar, maximum gainable point
        """
        n = len(arr)
        T = [[[0,0] for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            T[i][i][0] = arr[i]
        for L in range(1, n):
            for i in range(n - L):
                j = i + L
                if arr[i] + T[i+1][j][1] > arr[j] + T[i][j-1][1]:
                    T[i][j][0] = arr[i] + T[i+1][j][1]
                    T[i][j][1] = T[i+1][j][0]
                else:
                    T[i][j][0] = arr[j] + T[i][j-1][1]
                    T[i][j][1] = T[i][j-1][0]
        
        return T[0][-1][0]





#   -------------------------------------------------------------
if __name__ == '__main__':
    dp = DynamicProgramming()
    #   -----------------------------
    # Testing longest increasing subsequence
    print('------------------------------------------------')
    print('Running longest increasing subsequence')  
    u, v = dp.longest_increasing_subsequence([0, 2.3, 4, 5, -2, 7, 8])
    print(u, v)
    #   -----------------------------
    # Testing 0/1 Knapsack
    print('------------------------------------------------')
    print('Running 0/1 Knapsack') 
    max_val, items = dp.knapsack01([5,4,3,1], [7,5,4,1], 7) 
    print(max_val, items)
    #   -----------------------------
    #   -----------------------------
    # Testing unbounded Knapsack
    print('------------------------------------------------')
    print('Running unbounded Knapsack') 
    max_val, items = dp.unbounded_knapsack([1,3,4,5], [10,40,50,70], 8)
    print(max_val, items)
    max_val, items = dp.unbounded_knapsack([5,4,3,1], [7,5,4,1], 7) 
    print(max_val, items)
    #   -----------------------------
    # Testing egg dropping
    print('------------------------------------------------')
    print('Running egg dropping') 
    print(dp.egg_dropping(6,2))
    #   -----------------------------
    # Testing matrix chain multiplication
    print('------------------------------------------------')
    print('Running matrix chain multiplication') 
    matrices = [[2,3], [3,6], [6,4], [4,5]]
    #matrices = [[2,3], [3,2], [1,5], [5,1]]
    print(dp.matrix_chain_multiplication(matrices))
    #   -----------------------------
    # Testing Optimal game strategy
    print('------------------------------------------------')
    print('Running optimal game strategy') 
    a = [8, 15, 3, 7]
    print(dp.optimal_game_strategy(a))
    