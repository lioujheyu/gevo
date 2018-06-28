#!/usr/bin/env python3
import argparse
import subprocess
import json
import sys
import csv
import filecmp
from io import StringIO

from deap import creator
from deap import base
from deap import tools

sys.path.append('/home/jliou4/genetic-programming/cuda_evolve')
import irind
import evolve

class program(evolve.evolution):
    # Parameters
    log = open('debug_log', 'w')
    cudaPTX = 'a.ptx'

    def __init__(self, editf, kernel, bin, args="", timeout=30, fitness='time',
                 llvm_src_filename='cuda-device-only-kernel.ll',
                 compare_filename="compare.json"):
        self.kernels = kernel
        self.appBinary = bin
        self.appArgs = "" if args is None else args
        self.timeout = timeout
        self.fitness_function = fitness

        try:
            with open(editf, 'r') as f:
                self.edits = eval(f.read())
        except IOError:
            print("Edit File {} does not exist".format(editf))
            exit(1)

        try:
            with open(llvm_src_filename, 'r') as f:
                self.initSrcEnc = f.read().encode()
        except IOError:
            print("File {} does not exist".format(llvm_src_filename))
            exit(1)

        try:
            self.verifier = json.load(open(compare_filename))
        except IOError:
            print("File {} does not exist".format(compare_filename))
            exit(1)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", irind.llvmIRrep, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register('individual', creator.Individual, srcEnc=self.initSrcEnc)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

    def edittest(self):
        baseInd = creator.Individual(self.initSrcEnc)
        baseFits = [self.evaluate(baseInd)[0] for i in range(3)]
        baseFit = float(sum(baseFits)) / len(baseFits)

        self.pop = self.toolbox.population(n=len(self.edits))
        fitness = [None] * len(self.edits)
        for ind,edit,fits in zip(self.pop, self.edits, fitness):
            ind.edits = [edit]
            print("Evalute edit: {}".format(edit), end='', flush=True)
            if ind.update_from_edits() == False:
                print(": cannot compile")
                continue
            fits = [self.evaluate(ind)[0] for i in range(3)]
            if None in fits:
                print(": execution failed")
                continue
            fit = float(sum(fits)) / len(fits)
            improvement = baseFit/fit
            print(": {}. Improvement: {}".format(fit, improvement))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze the performance of mutation edits for CUDA kernel")
    parser.add_argument('-k', '--kernel', type=str, required=True,
        help="Target kernel functionof the given CUDA application. Use comma to separate kernels.")
    parser.add_argument('-e', '--edit', type=str, required=True,
        help="The edit file")
    parser.add_argument('-t', '--timeout', type=int, default=30,
        help="The timeout period to evaluate the CUDA application")
    parser.add_argument('-fitf', '--fitness_function', type=str, default='time',
        help="What is the target fitness for the evolution. Default ot execution time. Can be changed to power")
    parser.add_argument('binary',help="Binary of the CUDA application", nargs='?', default='a.out')
    parser.add_argument('args',help="arguments for the application binary", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    kernel = args.kernel.split(',')
    alyz = program(editf=args.edit, kernel=kernel, bin=args.binary, args=args.args, timeout=args.timeout, fitness=args.fitness_function)

    print("      Target CUDA program: {}".format(args.binary))
    print("Args for the CUDA program: {}".format(" ".join(args.args)))
    print("           Target kernels: {}".format(" ".join(kernel)))
    print("       Evaluation Timeout: {}".format(args.timeout))
    print("         Fitness function: {}".format(args.fitness_function))
    print("                Edit file: {}".format(args.edit))

    try:
        alyz.edittest()
    except KeyboardInterrupt:
        subprocess.run(['killall', args.binary])