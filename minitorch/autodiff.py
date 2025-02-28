from dataclasses import dataclass
from typing import Any, Iterable, Tuple
from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals = list(vals)
    original = vals[arg]

    vals[arg] = original - epsilon
    lower = f(*vals)

    vals[arg] = original + epsilon
    upper = f(*vals)

    return (upper - lower) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    results = [variable]
    travel_index = 0
    seen = set()

    while True:
        for var in results[travel_index].parents:
            if (not var.is_constant()) and (var.unique_id not in seen):
                results.append(var)
                seen.add(var.unique_id)
        travel_index += 1

        if travel_index >= len(results):
            break

    return results


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    variables = topological_sort(variable)
    vals = {variable.unique_id: deriv}

    for curr_var in variables:
        if curr_var.is_leaf():
            curr_var.accumulate_derivative(vals[curr_var.unique_id])
        else:
            derivs = curr_var.chain_rule(vals[curr_var.unique_id])
            for var, der in derivs:
                if var.unique_id in vals:
                    vals[var.unique_id] = vals[var.unique_id] + der
                else:
                    vals[var.unique_id] = der


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
