#!/usr/bin/env python3

import argparse
import subprocess
import random
import csv
import json
import re
import pathlib
import sys
import filecmp
from io import StringIO

# import matplotlib.pyplot as plt
# import networkx as nx
from deap import base
from deap import creator
from deap import tools

sys.path.append('/home/jliou4/genetic-programming/cuda_evolve')
import irind

# Run shorter is better
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", irind.llvmIRrep, fitness=creator.FitnessMin)

class evolution:
    # Parameters
    log = open('debug_log', 'w')
    cudaPTX = 'a.ptx'

    # Content
    pop = []
    generation = 0

    stats = {
        'valid':0, 'invalid':0, 'infinite':0,
        'maxFit':[], 'avgFit':[], 'minFit':[]
    }

    def __init__(self, kernel, bin, args="",
                 llvm_src_filename='cuda-device-only-kernel.ll',
                 compare_filename="compare.json",
                 CXPB=0.8, MUPB=0.1):
        self.CXPB = CXPB
        self.MUPB = MUPB
        self.kernels = kernel
        self.appBinary = bin
        self.appArgs = "" if args is None else args
        for i in range(0, len(self.kernels)):
            self.kernels[i] = self.kernels[i] + '('

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

        # tools initilization
        self.history = tools.History()
        self.toolbox = base.Toolbox()
        self.toolbox.register('mutate', self.mutLLVM)
        self.toolbox.register('mate', self.cxOnePointLLVM)
        self.toolbox.register('select', tools.selTournament, tournsize=2)
        self.toolbox.register('individual', creator.Individual, srcEnc=self.initSrcEnc)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        # Decorate the variation operators
        self.toolbox.decorate("mate", self.history.decorator)
        self.toolbox.decorate("mutate", self.history.decorator)

    def printGen(self, gen):
        print("-- Generation %s --" % gen)
        print("  Max %s" % self.stats['maxFit'][gen])
        print("  Avg %s" % self.stats['avgFit'][gen])
        print("  Min %s" % self.stats['minFit'][gen])

    def writeStage(self):
        pathlib.Path('stage').mkdir(exist_ok=True)
        if self.generation == 0:
            stageFileName = "stage/startedits.json"
        else:
            stageFileName = "stage/" + str(self.generation) + ".json"

        with open(stageFileName, 'w') as fp:
            count = 0
            allEdits = [ind.edits for ind in self.pop]
            json.dump(allEdits, fp, indent=2)

    def resultCompare(self, stdoutStr):
        src = stdoutStr if self.verifier['source'] == 'stdout' else self.verifier['source']
        golden = self.verifier['golden']

        if self.verifier['mode'] == 'string':
            return False if src.find(golden) == -1 else True
        elif self.verifier['mode'] == 'file':
            try:
                return filecmp.cmp(src, golden)
            except IOError:
                print("File {} or {} cannot be found".format(src, golden))

    def mutLLVM(self, individual):
        trial = 0
        # cut, replace, insert, swap
        operations = ['c', 'r', 'i', 's']
        while trial < individual.lineSize:
            trial = trial + 1
            line1 = random.randint(1, individual.lineSize)
            line2 = random.randint(1, individual.lineSize)
            while line1 == line2:
                line2 = random.randint(1, individual.lineSize)

            op = random.choice(operations)
            mut_command = ['llvm-mutate']
            if op == 'c':
                mut_command.extend(['-' + op, str(line1)])
            else:
                mut_command.extend(['-' + op, str(line1) + ',' + str(line2)])

            proc = subprocess.run(mut_command,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  input=individual.srcEnc)
            if(proc.returncode != 0 or
               proc.stderr.decode().find('mismatch') != -1):
                continue

            test_ind = creator.Individual(proc.stdout)
            fit = self.evaluate(test_ind)
            if fit[0] == 0:
                continue

            # read the uniqueID of the processed instructions
            for line in proc.stderr.decode().split('\n'):
                result = re.search('\w+ (U[0-9.irs]+)(,(U[0-9.irs]+))?', line)
                if result != None:
                    break

            if result == None:
                print(proc.stderr.decode(), file=sys.stderr)
                individual.ptx('error.ptx')
                print(*mut_command)
                raise Exception("Could not understand the result from llvm-mutate")

            if op == 'c':
                inst_UID = ('-'+op, result.group(1))
            else:
                inst_UID = ('-'+op, result.group(1) + ',' + result.group(3))

            individual.update(srcEnc=proc.stdout)
            individual.edits.append(inst_UID)
            individual.fitness.values = fit
            return individual,

        print("Cannot get mutant to be compiled in {} trials".format(individual.lineSize))
        return individual,

    def cxOnePointLLVM(self, ind1, ind2):
        # check whether they have the same starting edits(accessor)
        start_point = 0
        for edit1, edit2 in zip(ind1.edits, ind2.edits):
            if edit1 == edit2:
                start_point = start_point + 1
            else:
                break
        if (len(ind1.edits)-1) <= start_point and (len(ind2.edits)-1) <= start_point:
            print("s:{}, i:{}, i:{}. meaningless crossover".format(start_point, len(ind1.edits)-1, len(ind2.edits)-1),
                  file=self.log)
            # Exist no meaningful crossover
            return ind1, ind2

        point1 = start_point
        point2 = start_point
        while point1 == start_point and point2 == start_point:
            if len(ind1.edits) > start_point:
                point1 = random.randint(start_point, len(ind1.edits)-1)
            if len(ind2.edits) > start_point:
                point2 = random.randint(start_point, len(ind2.edits)-1)

        cmd1 = ind1.edits[:point1] + ind2.edits[point2:]
        cmd2 = ind2.edits[:point2] + ind1.edits[point1:]

        cmd1 = irind.rearrage(cmd1)
        cmd2 = irind.rearrage(cmd2)

        proc1 = subprocess.run(['llvm-mutate'] + [i for j in cmd1 for i in j],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               input=self.initSrcEnc)
        child1 = creator.Individual(proc1.stdout)
        print(cmd1, file=self.log, flush=True)
        print(proc1.stderr.decode(), file=self.log, flush=True)
        fit1 = [0]
        if proc1.returncode == 0:
            fit1 = self.evaluate(child1)
        if fit1[0] != 0:
            ind1.update(proc1.stdout)
            ind1.edits = cmd1
            ind1.fitness.values = fit1

        proc2 = subprocess.run(['llvm-mutate'] + [i for j in cmd2 for i in j],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               input=self.initSrcEnc)
        child2 = creator.Individual(proc2.stdout)
        print(cmd2, file=self.log, flush=True)
        print(proc2.stderr.decode(), file=self.log, flush=True)
        fit2 = [0]
        if proc2.returncode == 0:
            fit2 = self.evaluate(child2)
        if fit2[0] != 0:
            ind2.update(proc2.stdout)
            ind2.edits = cmd2
            ind2.fitness.values = fit2

        print('c', end='', flush=True)
        return ind1, ind2

    def evaluate(self, individual):
        # link
        individual.ptx(self.cudaPTX)
        with open('a.ll', 'w') as f:
            f.write(individual.srcEnc.decode())

        # proc = subprocess.Popen(['nvprof', '--csv', '-u', 'us',
        proc = subprocess.Popen(['/usr/local/cuda/bin/nvprof',
                                 '--unified-memory-profiling', 'off',
                                 '--csv',
                                 '-u', 'us',
                                 './' + self.appBinary] + self.appArgs,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        if self.resultCompare(program_output) == False:
            print('x', end='', flush=True)
            self.stats['invalid'] = self.stats['invalid'] + 1
            return 0,
        else:
            print('.', end='', flush=True)
            self.stats['valid'] = self.stats['valid'] + 1
            profile_output = stderr.decode()
            csv_list = list(csv.reader(StringIO(profile_output), delimiter=','))

            # search for kernel function(s)
            kernel_time = []
            for line in csv_list[5:]:
                # 8th column for name of CUDA function call
                for name in self.kernels:
                    # if line[7].split('(') == name:
                    if line[7].find(name) == 0:
                        # 3rd column for avg execution time
                        kernel_time.append(float(line[2]))

                if len(self.kernels) == len(kernel_time):
                    return sum(kernel_time),

            # kernel not found, remove '(' at the end of kernel name and find again
            kernels = [k[:-1] for k in self.kernels]
            for line in csv_list[5:]:
                for name in kernels:
                    if line[7].find(name) == 0:
                        kernel_time.append(float(line[2]))

                if len(kernels) == len(kernel_time):
                    return sum(kernel_time),

            raise Exception("{} is not a valid kernel function from nvprof".format(self.kernels))

    def evolve(self, resumeGen):
        if resumeGen == -1:
            print("Initialize the population. Size 100")
            self.pop = self.toolbox.population(n=100)

            # Initial 3x mutate to get diverse population
            for ind in self.pop:
                self.toolbox.mutate(ind)
                self.toolbox.mutate(ind)
                self.toolbox.mutate(ind)
        else:
            if resumeGen == 0:
                stageFileName = "stage/startedits.json"
            else:
                stageFileName = "stage/" + str(resumeGen) + ".json"

            try:
                allEdits = json.load(open(stageFileName))
            except:
                print(sys.exc_info())
                exit(1)

            print("Resume the population from {}. Size {}".format(stageFileName, len(allEdits)))
            self.pop = self.toolbox.population(n=len(allEdits))
            self.generation = resumeGen
            for edits, ind in zip(allEdits, self.pop):
                editsList = [(e[0], e[1]) for e in edits]
                ind.edits = editsList
                if ind.update_from_edits() == False:
                    raise Exception("Could not reconstruct ind from edits:{}".format(editsList))
                ind.fitness.values = self.evaluate(ind)

        popSize = len(self.pop)
        self.history.update(self.pop)

        fitnesses = [ind.fitness.values for ind in self.pop]
        fits = [ind.fitness.values[0] for ind in self.pop]

        self.stats['maxFit'].append(max(fits))
        self.stats['avgFit'].append((sum(fits)/len(fits)))
        self.stats['minFit'].append(min(fits))
        self.printGen(self.generation)

        # while generation < 100:
        while True:
            self.writeStage()
            offspring = self.toolbox.select(self.pop, popSize)
            # Preserve individual who has the highest fitness
            elite = tools.selBest(self.pop, 1)
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            elite = list(map(self.toolbox.clone, elite))
            with open("best-{}.ll".format(self.generation), 'w') as f:
                f.write(elite[0].srcEnc.decode())
            with open("best-{}.edit".format(self.generation), 'w') as f:
                print(elite[0].edits, file=f)

            self.generation = self.generation + 1

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if len(child1.edits) < 2 and len(child2.edits) < 2:
                    continue
                if random.random() < self.CXPB:
                    self.toolbox.mate(child1, child2)

            count = 0
            for mutant in offspring:
                if random.random() < self.MUPB:
                    count = count + 1
                    print(count, end='', flush=True)
                    del mutant.fitness.values
                    self.toolbox.mutate(mutant)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [self.evaluate(ind) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            self.pop[:] = offspring + elite
            # self.pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in self.pop]

            self.stats['maxFit'].append(max(fits))
            self.stats['avgFit'].append((sum(fits)/len(fits)))
            self.stats['minFit'].append(min(fits))
            print("")
            self.printGen(self.generation)

        # graph = nx.DiGraph(history.genealogy_tree)
        # graph = graph.reverse()     # Make the graph top-down
        # colors = [self.evaluate(history.genealogy_history[i])[0] for i in graph]
        # pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        # nx.draw(graph, pos, node_color=colors)
        # plt.savefig('genealogy.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evolve CUDA kernel function")
    parser.add_argument('-k', '--kernel', type=str, required=True,
        help="Target kernel functionof the given CUDA application. Use comma to separate kernels.")
    parser.add_argument('-r', '--resume', type=int, default=-1,
        help="Resume the process from genetating the population by reading stage/<RESUME>.json")
    parser.add_argument('binary',help="Binary of the CUDA application", nargs='?', default='a.out')
    parser.add_argument('args',help="arguments for the application binary", nargs='*')
    args = parser.parse_args()

    kernel = args.kernel.split(',')
    evo = evolution(kernel=kernel, bin=args.binary, args=args.args)

    print("      Target CUDA program: {}".format(args.binary))
    print("Args for the CUDA program: {}".format(" ".join(args.args)))
    print("           Target kernels: {}".format(" ".join(kernel)))

    try:
        evo.evolve(args.resume)
    except KeyboardInterrupt:
        print("valid variant:   {}".format(evo.stats['valid']))
        print("invalid variant: {}".format(evo.stats['invalid']))
        print("infinite variant:{}".format(evo.stats['infinite']))
        subprocess.run(['killall', args.binary])
