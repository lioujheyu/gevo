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

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", irind.llvmIRrep, fitness=creator.FitnessMin)

class program:
    # Parameters
    log = open('debug_log', 'w')
    cudaPTX = 'a.ptx'

    def __init__(self, editf, kernel, bin, args="", timeout=30,
                 llvm_src_filename='cuda-device-only-kernel.ll',
                 compare_filename="compare.json"):
        self.kernels = kernel
        self.appBinary = bin
        self.appArgs = "" if args is None else args
        self.timeout = timeout

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

        self.toolbox = base.Toolbox()
        self.toolbox.register('individual', creator.Individual, srcEnc=self.initSrcEnc)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)

    def resultCompare(self, stdoutStr):
        src = stdoutStr if self.verifier['source'] == 'stdout' else self.verifier['source']
        golden = self.verifier['golden']

        if self.verifier['mode'] == 'string':
            return False if src.find(golden) == -1 else True
        elif self.verifier['mode'] == 'file':
            result = True
            for s, g in zip(src, golden):
                try:
                    result = result & filecmp.cmp(s, g)
                except IOError:
                    print("File {} or {} cannot be found".format(src, golden))
            return result
        else:
            raise Exception("Unknown comparing mode \"{}\" from compare.json".format(
                self.verifier['mode']))

    def evaluate(self, individual):
        # link
        individual.ptx(self.cudaPTX)
        with open('a.ll', 'w') as f:
            f.write(individual.srcEnc.decode())

        proc = subprocess.Popen(['/usr/local/cuda/bin/nvprof',
                                 '--unified-memory-profiling', 'off',
                                 '--csv',
                                 '-u', 'us',
                                 './' + self.appBinary] + self.appArgs,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout) # second
            retcode = proc.poll()
            # retcode == 9: error is from testing program, not nvprof
            if retcode != 9 and retcode != 0:
                print(stderr.decode(), file=sys.stderr)
                raise Exception('nvprof error')
        except subprocess.TimeoutExpired:
            # Sometimes terminating nvprof will not terminate the underlying cuda program
            # if that program is corrupted. So issue the kill command to those cuda app first
            print('8', end='', flush=True)
            subprocess.run(['killall', self.appBinary])
            proc.kill()
            proc.wait()
            return 0,

        program_output = stdout.decode()
        if self.resultCompare(program_output) == False:
            print('x', end='', flush=True)
            return 0,
        else:
            profile_output = stderr.decode()
            csv_list = list(csv.reader(StringIO(profile_output), delimiter=','))

            # search for kernel function(s)
            kernel_time = []
            # The stats starts after 5th line
            for line in csv_list[5:]:
                for name in self.kernels:
                    # 8th column for name of CUDA function call
                    try:
                        if line[7].split('(')[0] == name:
                            # 3rd column for avg execution time
                            kernel_time.append(float(line[2]))
                    except:
                        print(stderr.decode(), file=sys.stderr)
                        exit()

                if len(self.kernels) == len(kernel_time):
                    return sum(kernel_time),

            raise Exception("{} is not a valid kernel function from nvprof".format(self.kernels))

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
            fit = float(sum(fits)) / len(fits)
            if fit == 0:
                print(": execution failed")
            else:
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
    parser.add_argument('binary',help="Binary of the CUDA application", nargs='?', default='a.out')
    parser.add_argument('args',help="arguments for the application binary", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    kernel = args.kernel.split(',')
    alyz = program(editf=args.edit, kernel=kernel, bin=args.binary, args=args.args, timeout=args.timeout)

    print("      Target CUDA program: {}".format(args.binary))
    print("Args for the CUDA program: {}".format(" ".join(args.args)))
    print("           Target kernels: {}".format(" ".join(kernel)))
    print("       Evaluation Timeout: {}".format(args.timeout))

    alyz.edittest()

    # try:
    #     evo.evolve(args.resume)
    # except KeyboardInterrupt:
    #     print("valid variant:   {}".format(evo.stats['valid']))
    #     print("invalid variant: {}".format(evo.stats['invalid']))
    #     print("infinite variant:{}".format(evo.stats['infinite']))
    #     subprocess.run(['killall', args.binary])
