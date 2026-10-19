"""Small independent mathematical references for the tinygrad CPU course."""
import math


def matmul(a, b):
    if not a or not b or not a[0] or not b[0] or any(len(r) != len(a[0]) for r in a) or any(len(r) != len(b[0]) for r in b):
        raise ValueError("nonempty rectangular matrices required")
    if len(a[0]) != len(b) or any(not math.isfinite(x) for matrix in (a,b) for row in matrix for x in row):
        raise ValueError("finite values and matching inner dimensions required")
    return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def softmax(row):
    if not row or any(not math.isfinite(x) for x in row):
        raise ValueError("finite nonempty row required")
    exp = [math.exp(x-max(row)) for x in row]
    return [x/sum(exp) for x in exp]


def causal_attention(q, k, v):
    if not q or len(q) != len(k) or len(k) != len(v):
        raise ValueError("equal nonempty sequence lengths required")
    for matrix in (q,k,v):
        if not matrix[0] or any(len(row)!=len(matrix[0]) for row in matrix) or any(not math.isfinite(x) for row in matrix for x in row):
            raise ValueError("finite rectangular matrices required")
    if len(q[0])!=len(k[0]):
        raise ValueError("query/key feature dimensions differ")
    scores = matmul(q, [list(col) for col in zip(*k)])
    output = []
    for i, row in enumerate(scores):
        weights = softmax([x/math.sqrt(len(q[0])) for x in row[:i+1]])
        output.append([sum(weights[j]*v[j][c] for j in range(i+1)) for c in range(len(v[0]))])
    return output


def rmsnorm(row, epsilon=1e-6):
    if not row or any(not math.isfinite(x) for x in row) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("finite nonempty row and positive epsilon required")
    scale = math.sqrt(sum(x*x for x in row)/len(row)+epsilon)
    return [x/scale for x in row]


def swiglu(value, gate):
    # Two already-projected branches: value * SiLU(gate), not value*sigmoid(gate).
    if len(value) != len(gate) or any(not math.isfinite(x) for x in [*value,*gate]):
        raise ValueError("equal finite vectors required")
    def sigmoid(x):
        return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))
    return [v*g*sigmoid(g) for v,g in zip(value,gate)]


def finite_gradient(function, values, epsilon=1e-4):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive epsilon required")
    answer = []
    for i in range(len(values)):
        plus, minus = list(values), list(values)
        plus[i] += epsilon;minus[i] -= epsilon
        answer.append((function(plus)-function(minus))/(2*epsilon))
    return answer
