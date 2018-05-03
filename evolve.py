#!/usr/bin/env python3

import argparse
import subprocess
import random
import csv
import json
import re
import pathlib
import sys
from io import StringIO

# import matplotlib.pyplot as plt
# import networkx as nx
from deap import base
from deap import creator
from deap import tools

# Run shorter is better
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# We need something mutable in python so that the mutation and crossover are able to modify the individual in-place
creator.create("Individual", bytearray, fitness=creator.FitnessMin, line_size=0, cmd=[])

class evolution:
    # Parameters
    log = open('debug_log', 'w')
    appBinary = 'a.out'
    cudaPTX = 'a.ptx'
    initSrcEnc = ""
    kernels = []
    CXPB = 0.0
    MUPB = 0.0

    # Content
    pop = []

    stats = {
        'valid':0, 'invalid':0, 'infinite':0,
        'maxFit':[], 'avgFit':[], 'minFit':[]
    }

    def __init__(self, kernel,
                 llvm_src_filename='cuda-device-only-kernel.ll',
                 CXPB=0.8, MUPB=0.1):
        self.CXPB = CXPB
        self.MUPB = MUPB
        self.kernels = kernel
        for i in range(0, len(self.kernels)):
            self.kernels[i] = self.kernels[i] + '('

        try:
            f = open(llvm_src_filename, 'r')
            self.initSrcEnc = f.read().encode()
        except IOError:
            print("File {} does not exist".format(llvm_src_filename))
            exit(1)

    @staticmethod
    def rearrage(cmd):
        c_cmd = [c for c in cmd if c[0] == '-c']
        r_cmd = [c for c in cmd if c[0] == '-r']
        i_cmd = [c for c in cmd if c[0] == '-i']
        s_cmd = [c for c in cmd if c[0] == '-s']

        cmd[:] = s_cmd + i_cmd + r_cmd + c_cmd
        return cmd

    @staticmethod
    def lineSize(individual):
        try:
            readline_proc = subprocess.run( ['llvm-mutate', '-I'],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            input=individual,
                                            check=True )
        except subprocess.CalledProcessError as err:
            print(err.stderr, file=sys.stderr)
            raise Exception('llvm-mutate error')

        return int(readline_proc.stderr.decode())

    def mutLLVM(self, individual):
        trial = 0
        # cut, replace, insert, swap
        operations = ['c', 'r', 'i', 's']
        while trial < individual.line_size:
            trial = trial + 1
            line1 = random.randint(1, individual.line_size)
            line2 = random.randint(1, individual.line_size)
            while line1 == line2:
                line2 = random.randint(1, individual.line_size)

            op = random.choice(operations)
            mut_command = ['llvm-mutate']
            if op == 'c':
                mut_command.extend(['-' + op, str(line1)])
            else:
                mut_command.extend(['-' + op, str(line1) + ',' + str(line2)])

            proc = subprocess.run(mut_command,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  input=individual)
            if(proc.returncode != 0 or
               proc.stderr.decode().find('mismatch') != -1 or
               proc.stderr.decode().find('no use') != -1):
                continue

            test_ind = creator.Individual(proc.stdout)
            fit = self.link_and_run(test_ind)
            if fit[0] == 0:
                continue

            # read the uniqueID of the processed instructions
            for line in proc.stderr.decode().split('\n'):
                result = re.search('\w+ (U[0-9.irs]+)(,(U[0-9.irs]+))?', line)
                if result != None:
                    break

            if result == None:
                print(proc.stderr.decode(), file=sys.stderr)
                raise Exception("Could not understand the result from llvm-mutate")

            if op == 'c':
                inst_UID = ('-'+op, result.group(1))
            else:
                inst_UID = ('-'+op, result.group(1) + ',' + result.group(3))

            individual[:] = bytearray(proc.stdout)
            individual.line_size = self.lineSize(individual)
            individual.cmd.append(inst_UID)
            individual.fitness.values = fit
            return individual,

        print("Cannot get mutant to be compiled in {} trials".format(individual.line_size))
        return individual,

    def cxOnePointLLVM(self, ind1, ind2):
        # check whether they have the same starting edits(accessor)
        start_point = 0
        for edit1, edit2 in zip(ind1.cmd, ind2.cmd):
            if edit1 == edit2:
                start_point = start_point + 1
            else:
                break
        if (len(ind1.cmd)-1) <= start_point and (len(ind2.cmd)-1) <= start_point:
            print("s:{}, i:{}, i:{}. meaningless crossover".format(start_point, len(ind1.cmd)-1, len(ind2.cmd)-1),
                  file=self.log)
            # Exist no meaningful crossover
            return ind1, ind2

        point1 = start_point
        point2 = start_point
        while point1 == start_point and point2 == start_point:
            if len(ind1.cmd) > start_point:
                point1 = random.randint(start_point, len(ind1.cmd)-1)
            if len(ind2.cmd) > start_point:
                point2 = random.randint(start_point, len(ind2.cmd)-1)

        cmd1 = ind1.cmd[:point1] + ind2.cmd[point2:]
        cmd2 = ind2.cmd[:point2] + ind1.cmd[point1:]
        # this set approach reduces the duplicate edits in the list
        cmd1[:] = list(set(cmd1))
        cmd2[:] = list(set(cmd2))
        # rearrage cmd to reduce the fail chance of edit
        self.rearrage(cmd1)
        self.rearrage(cmd2)

        proc1 = subprocess.run(['llvm-mutate'] + [i for j in cmd1 for i in j],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               input=self.initSrcEnc)
        child1 = creator.Individual(proc1.stdout)
        print(cmd1, file=self.log, flush=True)
        print(proc1.stderr.decode(), file=self.log, flush=True)
        fit1 = [0]
        if proc1.returncode == 0:
            fit1 = self.link_and_run(child1)
        if fit1[0] != 0:
            ind1[:] = bytearray(proc1.stdout)
            ind1.fitness.values = fit1
            ind1.line_size = self.lineSize(ind1)

        proc2 = subprocess.run(['llvm-mutate'] + [i for j in cmd2 for i in j],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               input=self.initSrcEnc)
        child2 = creator.Individual(proc2.stdout)
        print(cmd2, file=self.log, flush=True)
        print(proc2.stderr.decode(), file=self.log, flush=True)
        fit2 = [0]
        if proc2.returncode == 0:
            fit2 = self.link_and_run(child2)
        if fit2[0] != 0:
            ind2[:] = bytearray(proc2.stdout)
            ind2.fitness.values = fit2
            ind2.line_size = self.lineSize(ind2)

        print('c', end='', flush=True)
        return ind1, ind2

    def translate_llvmIR_ptx(self, llvmIR_str, outFileName="a.ptx"):
        proc = subprocess.run(['llc', '-o', outFileName],
                              stdout=subprocess.PIPE,
                              input=llvmIR_str)
        if proc.returncode is not 0:
            print(proc.stderr)
            raise Exception('llc error')

    def link_and_run(self, individual):
        # link
        self.translate_llvmIR_ptx(individual)
        with open('a.ll', 'w') as f:
            f.write(individual.decode())

        # proc = subprocess.Popen(['nvprof', '--csv', '-u', 'us',
        proc = subprocess.Popen(['/usr/local/cuda/bin/nvprof',
                                 '--unified-memory-profiling', 'off',
                                 '--csv',
                                 '-u', 'us',
                                 './' + self.appBinary],
                                stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            stdout, stderr = proc.communicate(timeout=30) # second
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
            self.stats['infinite'] = self.stats['infinite'] + 1
            return 0,

        program_output = stdout.decode()
        if program_output.find("Total Errors = 0") == -1:
        # if program_output.find("Test passed") == -1:
            print('x', end='', flush=True)
            self.stats['invalid'] = self.stats['invalid'] + 1
            return 0,
        else:
            print('.', end='', flush=True)
            self.stats['valid'] = self.stats['valid'] + 1
            profile_output = stderr.decode()
            f = StringIO(profile_output)
            csv_list = list(csv.reader(f, delimiter=','))

            # search for kernel function(s)
            kernel_time = []
            for line in csv_list[5:]:
                # 8th column for name of CUDA function call
                for name in self.kernels:
                    if line[7].find(name) == 0:
                        # 3rd column for avg execution time
                        kernel_time.append(float(line[2]))

                if len(self.kernels) == len(kernel_time):
                    return sum(kernel_time),

            raise Exception("{} is not a valid kernel function from nvprof".format(self.kernels))

    def readLLVMsrc(self, str_encode):
        I = creator.Individual(str_encode)
        I.line_size = self.lineSize(I)
        I.cmd = []
        return I

    def evolve(self):
        history = tools.History()
        toolbox = base.Toolbox()

        toolbox.register('evaluate', self.link_and_run)
        toolbox.register('mutate', self.mutLLVM)
        toolbox.register('mate', self.cxOnePointLLVM)
        toolbox.register('select', tools.selTournament, tournsize=2)
        toolbox.register('individual', self.readLLVMsrc)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        # Decorate the variation operators
        toolbox.decorate("mate", history.decorator)
        toolbox.decorate("mutate", history.decorator)

        self.pop = toolbox.population(n=100)
        popSize = len(self.pop)
        history.update(self.pop)
        # compile the initial llvm-IR into ptx and store it for later comparision
        self.translate_llvmIR_ptx(self.pop[0], "origin.ptx")

        # Initial 3x mutate to get diverse population
        for ind in self.pop:
            toolbox.mutate(ind)
            toolbox.mutate(ind)
            toolbox.mutate(ind)

        pathlib.Path('stage').mkdir(exist_ok=True)
        fp = open("stage/initcmd.json", 'w')
        count = 0
        all_cmd = [ind.cmd for ind in self.pop]
        json.dump(all_cmd, fp, indent=2)
        for ind in self.pop:
            filename = "stage/" + str(count) + ".ll"
            with open(filename, 'w') as f:
                f.write(ind.decode())
            count = count + 1
        fp.close()

        fitnesses = [ind.fitness.values for ind in self.pop]
        fits = [ind.fitness.values[0] for ind in self.pop]

        generations = 0

        print("-- Generation 0 --")
        self.stats['maxFit'].append(max(fits))
        self.stats['avgFit'].append((sum(fits)/len(fits)))
        self.stats['minFit'].append(min(fits))
        print("  Max %s" % self.stats['maxFit'][-1])
        print("  Avg %s" % self.stats['avgFit'][-1])
        print("  Min %s" % self.stats['minFit'][-1])

        # while generations < 100:
        while True:
            generations = generations + 1
            count = 0
            print("-- Generation %i --" % generations)

            offspring = toolbox.select(self.pop, popSize)
            # Preserve individual who has the highest fitness
            elite = tools.selBest(self.pop, 1)
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            elite = list(map(toolbox.clone, elite))
            with open("best-{}.ll".format(generations-1), 'w') as f:
                f.write(elite[0].decode())

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if len(child1.cmd) < 2 and len(child2.cmd) < 2:
                    continue
                if random.random() < self.CXPB:
                    toolbox.mate(child1, child2)

            for mutant in offspring:
                if random.random() < self.MUPB:
                    count = count + 1
                    print(count, end='', flush=True)
                    del mutant.fitness.values
                    toolbox.mutate(mutant)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [toolbox.evaluate(ind) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            self.pop[:] = offspring + elite
            # pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in self.pop]

            self.stats['maxFit'].append(max(fits))
            self.stats['avgFit'].append((sum(fits)/len(fits)))
            self.stats['minFit'].append(min(fits))
            print("")
            print("  Max %s" % self.stats['maxFit'][-1])
            print("  Avg %s" % self.stats['avgFit'][-1])
            print("  Min %s" % self.stats['minFit'][-1])

        # graph = nx.DiGraph(history.genealogy_tree)
        # graph = graph.reverse()     # Make the graph top-down
        # colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
        # pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        # nx.draw(graph, pos, node_color=colors)
        # plt.savefig('genealogy.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evolve CUDA kernel function")
    parser.add_argument('-k', '--kernel', help="Target kernel function. Use comma to separate kernels.")
    args = parser.parse_args()

    if args.kernel is None:
        print("Please specify the target kernel.",file=sys.stderr)
        parser.print_help()
        exit(1)

    kernel = args.kernel.split(',')
    evo = evolution(kernel)

    try:
        evo.evolve()
    except KeyboardInterrupt:
        print("valid variant:   {}".format(evo.stats['valid']))
        print("invalid variant: {}".format(evo.stats['invalid']))
        print("infinite variant:{}".format(evo.stats['infinite']))
