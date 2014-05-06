#!/usr/bin/env python3

"""
Test the speed of various least-squares regression implementations.

Usage:
  time_least_squares.py [options]

Options:
  -h --help           Show this screen.
  -r --repeats=<rp>   Number of repeats [default: 100]
  -t --min-time=<mt>  Shortest time that a run must take [default: 1]
  --test-N=<N>         The size of input to use for the test section [default: 100000]
"""

import docopt
import least_squares_regression
import numpy
import terminal_bars

from decimal import Decimal
from functools import partial
from math import floor, log10
from statistics import stdev
from timeit import Timer

options = docopt.docopt(__doc__)
N = int(options["--test-N"])
REPEATS = int(options["--repeats"])
MINTIME = float(options["--min-time"])

si_prefixes = [
    "y", "z", "a", "f", "p", "n", "µ", "m", "",
    "k", "M", "G", "T", "P", "E", "Z", "Y"
]

def engineering(number):
    number = Decimal("{:.3e}".format(number))
    exponents = floor(log10(number) / 3)

    return number.scaleb(-3*exponents), si_prefixes[exponents+8]

def format_constant_space(number, unit):
    scaled, prefix = engineering(number)
    return "{} {:<2}".format(scaled, prefix+unit)

def format_results(n, repeats, times):
    template = (
        "{n:8} items, {repeats} loops:  "
        "{mintime} (± {error})  per item  ({total} total)"
    )

    mintime = min(times) / (repeats*n)
    error = stdev(times) / (repeats*n)
    total = sum(times)

    return template.format(
        n=n, repeats=repeats,
        mintime = format_constant_space(mintime, "s"),
        error   = format_constant_space(error, "s"),
        total   = format_constant_space(total, "s")
    )


def orders_n(start=100, factor=2):
    x = start
    while True:
        x *= 2
        yield int(x)

functions = (
    least_squares_regression.matrix_lstsqr,
    least_squares_regression.auto_numpy_lstsqr,
    least_squares_regression.auto2_numpy_lstsqr,
    least_squares_regression.auto_scipy_lstsqr,
    least_squares_regression.untyped_lstsqr,
    least_squares_regression.simply_typed_lstsqr,
    least_squares_regression.memoryview_lstsqr,
    least_squares_regression.fully_typed_lstsqr,
    least_squares_regression.parallel_lstsqr,
)

function_times = {}
datasets = {}

numpy.random.seed(12345)
x = numpy.random.choice([0.8, 0.9, 1.0, 1.1], size=N) * numpy.arange(N)
y = numpy.random.choice([0.8, 0.9, 1.0, 1.1], size=N) * numpy.arange(N)
datasets[N] = x, y

print("TEST:")
print()

for function in functions:
    slope, intercept = function(*datasets[N])
    print("{:<20} y = {:.10f}·x + {:.10f}".format(function.__name__+":", slope, intercept))

print()
print()
print("TIME:")
print()

for function in functions:
    print(function.__name__)

    for N in orders_n():
        if N not in datasets:
            numpy.random.seed(12345)
            x = numpy.random.choice([0.8, 0.9, 1.0, 1.1], size=N) * numpy.arange(N)
            y = numpy.random.choice([0.8, 0.9, 1.0, 1.1], size=N) * numpy.arange(N)
            datasets[N] = x, y

        times = Timer(partial(function, *datasets[N])).repeat(int(REPEATS**0.5), REPEATS)
        function_times[function] = min(times) / (REPEATS*N)
        print(format_results(N, REPEATS, times), end="\r")

        if sum(times) > MINTIME:
            break

    print()
    print()

print()
print()
print("SUMMARY:")
print()

def simpleformatter(num):
    if num < 10:
        return str(round(num, 1))
    else:
        return str(round(num))

finaltimes = sorted(function_times.items(), key=lambda i: i[1])
besttime = finaltimes[0][1]

names = [function.__name__ for function, _ in finaltimes]
times = [time / besttime   for _, time     in finaltimes]

terminal_bars.plot(names, times, 100, formatter=simpleformatter)

print()
print("Zoomed:")
print()

terminal_bars.plot(names, times, 100, formatter=simpleformatter, maximum=times[0]*20)