"""QP 公共接口：保留求解状态、热启动与约束残差，不静默使用失败结果。"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
import osqp


@dataclass
class QPInfo:
    status: str = "idle"
    seconds: float = 0.
    violation: float = 0.
    iterations: int = 0


class QPFailure(RuntimeError):
    pass


class QPSolver:
    def __init__(self, name):
        self.name = name
        self.solver = None
        self.pattern = None
        self.last_x = None
        self.info = QPInfo()

    def solve(self, P, q, A, lower, upper):
        P = sparse.triu(sparse.csc_matrix(P), format="csc")
        A = sparse.csc_matrix(A)
        P.sort_indices()
        A.sort_indices()
        pattern = (P.indptr.tobytes(), P.indices.tobytes(), A.indptr.tobytes(), A.indices.tobytes())
        if self.pattern != pattern:
            # 接触切换可能改变稀疏结构；这时重建问题，但仍保留可用的原变量初值。
            self.solver = osqp.OSQP()
            self.solver.setup(P=P, q=q, A=A, l=lower, u=upper, verbose=False,
                              eps_abs=1e-4, eps_rel=1e-5, max_iter=6000,
                              polishing=True, warm_starting=True)
            if self.last_x is not None and len(self.last_x) == len(q):
                self.solver.warm_start(x=self.last_x)
            self.pattern = pattern
        else:
            self.solver.update(Px=P.data, Ax=A.data, q=q, l=lower, u=upper)
        result = self.solver.solve(raise_error=False)
        violation = np.inf
        if result.info.status_val in (1, 2) and result.x is not None and np.isfinite(result.x).all():
            value = A @ result.x
            violation = float(max(0., np.max(lower-value), np.max(value-upper)))
        self.info = QPInfo(result.info.status, result.info.run_time, violation, result.info.iter)
        if result.info.status_val not in (1, 2) or violation > 0.02:
            raise QPFailure(f"{self.name}: {self.info}")
        self.last_x = result.x.copy()
        return result.x


def force_constraints(contact, friction, maximum):
    """保守摩擦棱锥：|fx|+|fy| <= μ fz，0 <= fz <= fmax。

    它内接于圆形库仑摩擦锥。摆动腿把 fmax 设为零，三方向力均被约束为零。
    """
    C = np.zeros((20, 12))
    lower, upper = np.full(20, -np.inf), np.zeros(20)
    block = np.array([[1, 1, -friction], [1, -1, -friction],
                      [-1, 1, -friction], [-1, -1, -friction], [0, 0, 1.]])
    for leg in range(4):
        C[5*leg:5*leg+5, 3*leg:3*leg+3] = block
        lower[5*leg+4] = 0.
        upper[5*leg+4] = maximum if contact[leg] else 0.
    return sparse.csc_matrix(C), lower, upper
