"""Deterministic maths helpers for CBSE Study Assistant.

Handles fraction-to-percentage conversions, arithmetic, HCF/LCM, simple
one-variable equations, and balancing simple neutral chemical equations.
Returns an empty string when a question is not handled so normal RAG can run.
"""
import math
import re
from functools import reduce
from fractions import Fraction

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
    _TRANSFORMS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
except Exception:
    sp = None
    _TRANSFORMS = None

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def _norm(text):
    text = str(text or "")
    for old, new in (
        ("×", "*"), ("÷", "/"), ("−", "-"), ("–", "-"), ("—", "-"),
        ("²", "^2"), ("³", "^3"), ("→", "->"), ("⟶", "->"), ("⟹", "->"),
    ):
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def _fmt(value):
    if sp is None:
        return str(value)
    if getattr(value, "is_Float", False) and float(value).is_integer():
        value = sp.nsimplify(value)
    if getattr(value, "is_Integer", False):
        return str(int(value))
    if getattr(value, "is_Rational", False):
        return f"{value} (= {float(value):.6g})"
    return str(sp.N(value, 10))


def _fraction_percentage(question):
    """Solve a fraction-to-percentage question exactly, without an LLM."""
    q = _norm(question)
    if not re.search(r"\bpercentage\b|%", q, re.IGNORECASE):
        return ""

    # Prefer a fraction written with a slash, e.g. 7/14 or 7 / 14.
    match = re.search(r"(?<![\w.])(\d{1,12})\s*/\s*(\d{1,12})(?!\d)", q)
    if not match:
        # Also support wording such as "fraction 7 over 14 as a percentage".
        match = re.search(
            r"\b(?:fraction\s+)?(\d{1,12})\s+over\s+(\d{1,12})\b", q, re.I
        )
    if not match:
        return ""

    numerator, denominator = int(match.group(1)), int(match.group(2))
    if denominator == 0:
        return ""

    value = Fraction(numerator * 100, denominator)
    if value.denominator == 1:
        result = str(value.numerator)
    else:
        # Decimal output is for display only; Fraction keeps the calculation exact.
        decimal_value = value.numerator / value.denominator
        result = f"{decimal_value:.10f}".rstrip("0").rstrip(".")
    return f"{numerator}/{denominator} × 100 = {result}%\nAnswer: {result}%"


_CMD = re.compile(
    r"^(?:please\s+)?(?:calculate|compute|evaluate|simplify|find(?:\s+the\s+value\s+of)?"
    r"|what\s+is|what's|value\s+of)\s+",
    re.I,
)


def _arith(question):
    if sp is None:
        return ""
    expression = _CMD.sub("", question).rstrip(" ?.=!")
    if len(expression) > 80 or not re.fullmatch(r"[0-9+\-*/().^ ]+", expression):
        return ""
    if not re.search(r"\d\s*[-+*/^]\s*[\d(]", expression):
        return ""
    if any(int(power) > 100 for power in re.findall(r"\^\s*(\d+)", expression)):
        return ""
    try:
        value = parse_expr(expression, transformations=_TRANSFORMS)
    except Exception:
        return ""
    if getattr(value, "free_symbols", None):
        return ""
    return f"{expression} = {_fmt(value)}"


_EQ_START = re.compile(
    r"^(?:please\s+)?(?:solve|find\s+the\s+(?:roots?|solutions?|zeros?)\s+of"
    r"|find\s+(?:the\s+value\s+of\s+)?[a-z]\s+(?:if|when|from|in))\s+"
    r"(?:the\s+)?(?:(?:linear|quadratic)\s+)?(?:equation\s*)?[:,]?\s*(.+)$",
    re.I,
)


def _equation(question):
    if sp is None:
        return ""
    match = _EQ_START.match(question)
    if not match:
        return ""
    equation = match.group(1).strip().rstrip(" ?.!").replace("^", "**")
    if equation.count("=") != 1 or len(equation) > 80:
        return ""
    if not re.fullmatch(r"[0-9a-zA-Z+\-*/().^= ]+", equation):
        return ""
    if re.search(r"[A-Za-z]{2,}", equation):
        return ""
    variables = sorted(set(re.findall(r"[A-Za-z]", equation)))
    if len(variables) != 1:
        return ""
    name = variables[0]
    variable = sp.Symbol(name)
    left, right = (part.strip() for part in equation.split("="))
    try:
        lhs = parse_expr(left, transformations=_TRANSFORMS, local_dict={name: variable})
        rhs = parse_expr(right, transformations=_TRANSFORMS, local_dict={name: variable})
        solutions = sp.solve(sp.Eq(lhs, rhs), variable)
    except Exception:
        return ""
    if any("RootOf" in str(solution) for solution in solutions):
        return ""
    real_solutions = []
    for solution in solutions:
        try:
            if solution.is_real is True or sp.im(solution) == 0:
                real_solutions.append(solution)
        except Exception:
            continue
    if not real_solutions:
        return f"{left} = {right} has no real solution."
    try:
        real_solutions.sort(key=lambda solution: float(solution))
    except (TypeError, ValueError):
        return ""
    lines = [
        f"{left} = {right}",
        "Solution: " + " or ".join(f"{name} = {_fmt(solution)}" for solution in real_solutions),
    ]
    for solution in real_solutions:
        lines.append(
            f"Check: for {name} = {_fmt(solution)}, "
            f"LHS = {_fmt(sp.nsimplify(lhs.subs(variable, solution)))} "
            f"and RHS = {_fmt(sp.nsimplify(rhs.subs(variable, solution)))}."
        )
    return "\n".join(lines)


def _hcf_lcm(question):
    lower = question.lower()
    want_hcf = re.search(
        r"\b(hcf|gcd|highest common factor|greatest common (?:divisor|factor))\b",
        lower,
    )
    want_lcm = re.search(r"\b(lcm|least common multiple|lowest common multiple)\b", lower)
    if not (want_hcf or want_lcm) or len(lower.split()) > 25 or "prime factor" in lower:
        return ""
    match = re.search(r"\b(?:of|for|between)\s+(\d+(?:\s*(?:,|and|&)\s*\d+)+)", lower)
    if not match:
        return ""
    numbers = [int(number) for number in re.findall(r"\d+", match.group(1))]
    if not 2 <= len(numbers) <= 6 or any(number < 1 or number > 10**12 for number in numbers):
        return ""
    if "euclid" in lower and want_hcf and len(numbers) == 2:
        a, b = max(numbers), min(numbers)
        lines = [f"Using Euclid's division algorithm to find HCF({a}, {b}):"]
        x, y = a, b
        while y:
            lines.append(f"{x} = {y} × {x // y} + {x % y}")
            x, y = y, x % y
        lines.extend([
            f"The remainder has become 0, so the last divisor {x} is the HCF.",
            f"HCF({a}, {b}) = {x}",
        ])
        if want_lcm:
            lines.append(f"LCM({a}, {b}) = ({a} × {b}) ÷ {x} = {a * b // x}")
        return "\n".join(lines)
    hcf = reduce(math.gcd, numbers)
    lcm = reduce(lambda a, b: a * b // math.gcd(a, b), numbers)
    label = ", ".join(map(str, numbers))
    results = []
    if want_hcf:
        results.append(f"HCF({label}) = {hcf}")
    if want_lcm:
        results.append(f"LCM({label}) = {lcm}")
    return ". ".join(results) + "."


def _balance_chemical_equation(question):
    """Balance simple neutral formulas; intentionally skips brackets and ionic charges."""
    if sp is None or not re.search(r"\b(balance|balancing)\b", question, re.I):
        return ""
    normalized = _norm(question)
    equation = re.sub(
        r"^.*?\b(?:balance|balancing)\s+(?:the\s+)?(?:simple\s+)?"
        r"(?:chemical\s+)?equation\s*[:\-]?\s*",
        "",
        normalized,
        flags=re.I,
    ).strip()
    if not re.search(r"(?:->|=)", equation):
        return ""
    equation = equation.strip().rstrip(" ?.!").replace("→", "->")
    parts = re.split(r"\s*(?:->|=)\s*", equation, maxsplit=1)
    if len(parts) != 2:
        return ""
    left, right = (part.strip() for part in parts)

    def parse_side(side):
        formulas = [item.strip() for item in re.split(r"\s*\+\s*", side)]
        if not formulas or any(not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", item) for item in formulas):
            return None
        return formulas

    reactants, products = parse_side(left), parse_side(right)
    if not reactants or not products:
        return ""

    def atom_counts(formula):
        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        if not tokens or "".join(element + count for element, count in tokens) != formula:
            return None
        counts = {}
        for element, count in tokens:
            counts[element] = counts.get(element, 0) + int(count or "1")
        return counts

    formulas = reactants + products
    parsed = [atom_counts(formula) for formula in formulas]
    if any(counts is None for counts in parsed):
        return ""
    elements = sorted(set().union(*(counts.keys() for counts in parsed)))
    matrix = sp.Matrix([
        [
            (1 if index < len(reactants) else -1) * counts.get(element, 0)
            for index, counts in enumerate(parsed)
        ]
        for element in elements
    ])
    try:
        basis = matrix.nullspace()
        if len(basis) != 1:
            return ""
        vector = basis[0]
        multiplier = sp.ilcm(*[int(value.q) for value in vector])
        coefficients = [int(value * multiplier) for value in vector]
        common = reduce(math.gcd, [abs(value) for value in coefficients])
        if common == 0:
            return ""
        coefficients = [value // common for value in coefficients]
        if any(value <= 0 for value in coefficients):
            return ""

        def render(coefficient, formula):
            return f"{coefficient} {formula}" if coefficient != 1 else formula

        lhs = " + ".join(
            render(coefficient, formula)
            for coefficient, formula in zip(coefficients[:len(reactants)], reactants)
        )
        rhs = " + ".join(
            render(coefficient, formula)
            for coefficient, formula in zip(coefficients[len(reactants):], products)
        )
        return f"Balanced equation: {lhs} → {rhs}"
    except Exception:
        return ""


def solve_math(question):
    """Return an exact deterministic answer, or '' when normal RAG should handle it."""
    if not question or _DEVANAGARI.search(question):
        return ""
    normalized = _norm(question)
    if not normalized or len(normalized) > 200:
        return ""
    for function in (
        _fraction_percentage,
        _balance_chemical_equation,
        _hcf_lcm,
        _equation,
        _arith,
    ):
        try:
            result = function(normalized)
        except Exception:
            result = ""
        if result:
            return result
    return ""
