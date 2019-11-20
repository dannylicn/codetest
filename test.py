import numpy as np

def unbounded_knapsack(w, val, W):
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
        print(np.matrix(score_mat))
        for i in range(1, len(arr)):
            for j in range(1, W+1):
                if j < arr[i][1]:
                    score_mat[i][j] = score_mat[i-1][j]
                else:
                    score_mat[i][j] = max(score_mat[i-1][j], 
                                        arr[i][0] + score_mat[i][j - arr[i][1]]
                                        )
        print(np.matrix(score_mat))
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

u, v = unbounded_knapsack([1,3,4,5], [10,40,50,70], 8)
print(u, v)