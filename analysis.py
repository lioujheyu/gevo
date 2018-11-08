#!/usr/bin/env python3
import argparse
import subprocess
import json
import sys
import csv
import filecmp
from io import StringIO
from itertools import cycle

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

    def __init__(self, editf, kernel, bin, profile, timeout=30, fitness='time',
                 llvm_src_filename='cuda-device-only-kernel.ll', err_rate=0.01):
        self.kernels = kernel
        self.appBinary = bin
        # self.appArgs = "" if args is None else args
        self.timeout = timeout
        self.fitness_function = fitness
        self.err_rate = err_rate

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

        self.verifier = profile['verify']

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", irind.llvmIRrep, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register('individual', creator.Individual, srcEnc=self.initSrcEnc)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

        # Set up testcase
        self.origin = creator.Individual(self.initSrcEnc)
        self.origin.ptx(self.cudaPTX)
        arg_array = [[]]
        for i, arg in enumerate(profile['args']):
            if arg.get('bond', None) is None:
                arg_array_next = [ e[:] for e in arg_array for _ in range(len(arg['value']))]
                arg_array = arg_array_next
                for e1, e2 in zip(arg_array, cycle(arg['value'])):
                    e1.append(e2)
            else:
                for e in arg_array:
                    bonded_arg = arg['bond'][0]
                    bonded_idx = profile['args'][bonded_arg]['value'].index(e[bonded_arg])
                    e.append(arg['value'][bonded_idx])

        arg_array = [ [str(e) for e in args ] for args in arg_array ]

        self.testcase = []
        for i in range(len(arg_array)):
            self.testcase.append( self._testcase(self, i, kernel, bin, profile['verify']) )
        print("evalute testcase as golden..", end='', flush=True)
        for i, (tc, arg) in enumerate(zip(self.testcase, arg_array)):
            tc.args = arg
            print("{}..".format(i+1), end='', flush=True)
            tc.evaluate()
        print("done", flush=True)

        fits = [ tc.fitness[0] for tc in self.testcase]
        errs = [ tc.fitness[1] for tc in self.testcase]
        self.origin.fitness.values = (sum(fits)/len(fits), max(errs))

    def edittest(self):
        self.pop = self.toolbox.population(n=len(self.edits))
        fitness = [None] * len(self.edits)
        for ind,edit,fits in zip(self.pop, self.edits, fitness):
            ind.edits = [edit]
            print("Evalute edit: {}".format(edit), end='', flush=True)
            if ind.update_from_edits() == False:
                print(": cannot compile")
                continue
            fits = [self.evaluate(ind)[0] for i in range(3)]
            errs = [self.evaluate(ind)[1] for i in range(3)]
            if None in fits:
                print(": execution failed")
                continue
            fit = float(sum(fits)) / len(fits)
            err = float(sum(errs)) / len(errs)
            improvement = self.origin.fitness.values[0]/fit
            print(": {}. Improvement: {}. Error:{}".format(fit, improvement, err))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze the performance of mutation edits for CUDA kernel")
    parser.add_argument('-P', '--profile_file', type=str, required=True,
        help="Specify the profile file that contains all application execution and testing information")
    parser.add_argument('-e', '--edit', type=str, required=True,
        help="The edit file")
    parser.add_argument('-t', '--timeout', type=int, default=30,
        help="The timeout period to evaluate the CUDA application")
    parser.add_argument('-fitf', '--fitness_function', type=str, default='time',
        help="What is the target fitness for the evolution. Default ot execution time. Can be changed to power")
    parser.add_argument('--err_rate', type=float, default='0.01',
        help="Allowed maximum relative error generate from mutant comparing to the origin")
    args = parser.parse_args()

    try:
        profile = json.load(open(args.profile_file))
    except:
        print(sys.exc_info())
        exit(-1)

    alyz = program(
        editf=args.edit,
        kernel=profile['kernels'],
        bin=profile['binary'],
        profile=profile,
        timeout=args.timeout,
        fitness=args.fitness_function,
        err_rate=args.err_rate)

    print("      Target CUDA program: {}".format(profile['binary']))
    print("Args for the CUDA program:")
    for tc in alyz.testcase:
        print("\t{}".format(" ".join(tc.args)))
    print("           Target kernels: {}".format(" ".join(profile['kernels'])))
    print("       Evaluation Timeout: {}".format(args.timeout))
    print("         Fitness function: {}".format(args.fitness_function))
    print("                Edit file: {}".format(args.edit))
    print("      Tolerate Error Rate: {}".format(args.err_rate))

    try:
        alyz.edittest()
    except KeyboardInterrupt:
        subprocess.run(['killall', args.binary])
