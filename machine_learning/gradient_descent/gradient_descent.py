class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        
        grad_fn = 2 * init

        i = 0
        while i < iterations:
            
            init = init - learning_rate * grad_fn

            grad_fn = 2 * init

            i += 1
        
        return round(init, 5)
