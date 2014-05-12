#!/usr/bin/env python3

"""
Test the speed of various least-squares regression implementations.

Usage:
  time_least_squares.py [options]

Options:
  -h --help           Show this screen.
  -r --repeats=<rp>   Number of repeats. Minimum 4. [default: 16]
  -t --min-time=<mt>  Shortest time that a run must take [default: 1]
  --test-N=<N>         The size of input to use for the test section [default: 100000]
"""

import docopt
import least_squares_regression
import least_squares_regression_bytecode
import numpy
import terminal_bars

from blessings import Terminal
from decimal import Decimal
from functools import partial
from math import isnan, floor, log10
from statistics import stdev, StatisticsError
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
    if isnan(number):
        return "???.? {} ".format(unit)

    scaled, prefix = engineering(number)
    return "{} {:<2}".format(scaled, prefix+unit)

def format_results(n, repeats, times, completed=1):
    template = (
        "{n:10} items, {repeats} loops:  "
        "{mintime} (± {error})  per item  "
        "({total} total) [{percent:4.0%}]"
    )

    mintime = min(times) / (repeats*n)

    try:
        # True time is in theory below the lowest time,
        # so all deviation is above it
        error = stdev(times, xbar=mintime) / (repeats*n)
    except StatisticsError:
        error = float("nan")

    total = sum(times)

    return template.format(
        n=n, repeats=repeats,
        mintime = format_constant_space(mintime, "s"),
        error   = format_constant_space(error, "s"),
        total   = format_constant_space(total, "s"),
        percent = completed
    )

def orders_n(start=100, factor=2):
    x = start
    while True:
        x *= 2
        yield int(x)

functions = (
    least_squares_regression_bytecode.bytecode_matrix_lstsqr,
    least_squares_regression_bytecode.bytecode_auto_numpy_lstsqr,
    least_squares_regression_bytecode.bytecode_auto2_numpy_lstsqr,
    least_squares_regression_bytecode.bytecode_auto_scipy_lstsqr,
    least_squares_regression_bytecode.bytecode_untyped_lstsqr,
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

function_times = dict.fromkeys(functions, float("inf"))
datasets = {}

numpy.random.seed(12345)
x = numpy.random.choice([0.8, 0.9, 1.0, 1.1], size=N) * numpy.arange(N)
y = numpy.random.choice([0.8, 0.9, 1.0, 1.1], size=N) * numpy.arange(N)
datasets[N] = x, y

print("TEST:")
print()

namespace = max(len(function.__name__) for function in functions) + 1

for function in functions:
    slope, intercept = function(*datasets[N])
    print("{:<{}} y = {:.10f}·x + {:.10f}".format(function.__name__+":", namespace, slope, intercept))

print()
print()
print("TIME:")
print()



def print_summary():
    print()
    print()
    print("SUMMARY:")
    print()

    def simpleformatter(num):
        if num < 10:
            return str(round(num, 1))

        else:
            try:
                return str(round(num))
            except (ValueError, OverflowError):
                return "NaN"

    finaltimes = sorted(function_times.items(), key=lambda i: i[1])
    besttime = finaltimes[0][1]

    names = [function.__name__ for function, _ in finaltimes]
    times = [time / besttime   for _, time     in finaltimes]

    terminal_bars.plot(names, times, 200, formatter=simpleformatter)

    print()
    print("Zoomed:")
    print()

    terminal_bars.plot(names, times, 200, formatter=simpleformatter, maximum=times[0]*20)


terminal = Terminal()
space_needed = len(functions)*2 + 12

import itertools
from heapq import heappush, heappop

pq = []                         # list of entries arranged in a heap
entry_finder = {}               # mapping of tasks to entries
REMOVED = object()              # placeholder for a removed task
counter = itertools.count()     # unique sequence count

def add_task(task, priority=0):
    'Add a new task or update the priority of an existing task'
    if task in entry_finder:
        remove_task(task)
    count = next(counter)
    entry = [priority, count, task]
    entry_finder[task] = entry
    heappush(pq, entry)

def remove_task(task):
    'Mark an existing task as REMOVED.  Raise KeyError if not found.'
    entry = entry_finder.pop(task)
    entry[-1] = REMOVED

def pop_task():
    'Remove and return the lowest priority task. Raise KeyError if empty.'
    while pq:
        priority, count, task = heappop(pq)
        if task is not REMOVED:
            del entry_finder[task]
            return task
    raise KeyError('pop from an empty priority queue')

for function in functions:
    add_task((function, orders_n()), 0)


for _ in range(space_needed):
    print()

while pq:
    function, ngenerator = pop_task()
    N = next(ngenerator)

    if N not in datasets:
        numpy.random.seed(12345)
        x = numpy.random.choice([0.8, 0.9, 1.0, 1.1], size=N) * numpy.arange(N)
        y = numpy.random.choice([0.8, 0.9, 1.0, 1.1], size=N) * numpy.arange(N)
        datasets[N] = x, y

    numtimes = int(REPEATS ** 0.5)
    times = []
    functimer = Timer(partial(function, *datasets[N]))

    for i in range(numtimes):
        times.append(functimer.timeit(REPEATS))

    for _ in range(space_needed):
        print(terminal.move_up(), end="")

    print("{:>30}   {}   {}".format(
        function.__name__,
        format_constant_space(N, ""),
        format_constant_space(sum(times), "s")
    ))
    print_summary()

    function_times[function] = min(times) / (REPEATS*N)

    if sum(times) < MINTIME:
        add_task((function, ngenerator), sum(times))
