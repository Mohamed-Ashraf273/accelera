class MergeWrapper:
    def __init__(self, merge_func, **params):
        if not callable(merge_func):
            raise ValueError("merge_func must be callable")

        self.merge_func = merge_func
        self.params = params

    def execute(self, branch_results):
        try:
            if self.params:
                return self.merge_func(branch_results, **self.params)
            else:
                return self.merge_func(branch_results)
        except Exception as e:
            raise RuntimeError(
                f"Error executing merge function: {str(e)}"
            ) from e

    def __call__(self, branch_results):
        return self.execute(branch_results)
