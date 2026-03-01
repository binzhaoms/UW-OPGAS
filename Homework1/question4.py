"""Compute logistic map iterates for different rho values.

Copied from earlier work, placed in question4 per user request.
"""

import pprint

def iterate_logistic(rho: float, x1: float, n: int) -> list[float]:
    """Return n iterates of the logistic map starting at x1."""
    x = x1
    result = []
    for _ in range(n):
        result.append(x)
        x = rho * x * (1 - x)
    return result


def main() -> None:
    rhos = [0.8, 1.5, 2.8, 3.2, 3.5, 3.65]
    x1 = 0.5
    n = 50
    # create list of iterates for each rho in the same order
    vectors = [iterate_logistic(rho, x1, n) for rho in rhos]

    # print header row with rho values
    header = "    ".join(f"rho={rho:>4}" for rho in rhos)
    print(header)
    print("" + "------" * len(rhos))

    # print rows: x0..x49 in columns
    for i in range(n):
        row_values = [(vectors[j][i]) for j in range(len(rhos))]
        # format each value to a fixed width for alignment
        formatted = "    ".join(f"{val:.6f}" for val in row_values)
        print(formatted)


if __name__ == "__main__":
    main()
