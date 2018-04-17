#!/usr/bin/env python3

import argparse
import subprocess
import shlex
import pipes
import random
import sys
from io import StringIO
from deap import base
from deap import creator
from deap import tools
import csv
import json
import matplotlib.pyplot as plt
import networkx as nx

cuda_flags = shlex.split('-L/usr/local/cuda/lib64 --cuda-gpu-arch=sm_35 -ldl -lrt -pthread -lcudart_static -lcuda')

# Run shorter is better
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# We need something mutable in python so that the mutation and crossover are able to modify the individual in-place
creator.create("Individual", bytearray, fitness=creator.FitnessMin, line_size=0, cmd=[])

def lineSize(individual):
    try:
        readline_proc = subprocess.run( ['llvm-mutate', '-I'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        input=individual,
                                        check=True )
    except subprocess.CalledProcessError as err:
        print (err.stderr)
        raise Exception('llvm-mutate error')

    return int(readline_proc.stderr.decode())

def mutLLVM(individual, kernel, stats):
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
            inst = ['-' + op, str(line1)]
        else:
            inst = ['-' + op, str(line1) + ',' + str(line2)]
        mut_command.extend(inst)

        proc = subprocess.run( mut_command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               input=individual )
        if proc.returncode != 0:
            continue
        if proc.stderr.decode().find('mismatch') != -1:
            continue
        test_ind = creator.Individual(proc.stdout)
        fit = link_and_run(test_ind, kernel, stats)
        if fit[0] > 1000000000:
            continue

        individual[:] = bytearray(proc.stdout)
        individual.line_size = lineSize(individual)
        individual.cmd.append(inst)
        individual.fitness.values = fit
        return individual,

    print("Cannot get mutant to be compiled in {} trials".format(individual.line_size))
    return individual,

def cxOnePointLLVM(ind1, ind2, init_src):
    src_length = min(len(ind1.cmd), len(ind2.cmd))
    point = random.randint(1, src_length-1)
    cmd1 = ind1.cmd[:point] + ind2.cmd[point:]
    cmd2 = ind2.cmd[:point] + ind1.cmd[point:]
    # ind1.cmd = cmd1
    # ind2.cmd = cmd2
    # cmd1 = ['llvm-mutate']
    # cmd1.extend(ind1.cmd)
    # cmd2 = ['llvm-mutate']
    # cmd2.extend(ind2.cmd)
    proc1 = subprocess.run( ['llvm-mutate'] + [i for j in cmd1 for i in j],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            input=init_src )
    if proc1.returncode is 0:
        # if proc1.stderr.decode().find('mismatch') > 0:
        #     continue
        ind1[:] = bytearray(proc1.stdout)
        ind1.line_size = lineSize(ind1)

    proc2 = subprocess.run( ['llvm-mutate'] + [i for j in cmd2 for i in j],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            input=init_src )
    if proc2.returncode is 0:
        # if proc2.stderr.decode().find('mismatch') > 0:
        #     continue
        ind2[:] = bytearray(proc2.stdout)
        ind2.line_size = lineSize(ind2)

    print('c', end='', flush=True)
    return ind1, ind2


def translate_llvmIR_ptx(llvmIR_str, outFileName="a.ptx"):
    proc = subprocess.run( ['llc', '-o', outFileName],
                           stdout=subprocess.PIPE,
                           input=llvmIR_str )
    if proc.returncode is not 0:
        print (proc.stderr)
        raise Exception('llc error')

def link_and_run(individual, kernel, stats):
    cudaAppName = 'a.out'

    # link
    translate_llvmIR_ptx(individual)
    with open('a.ll', 'w') as f:
        f.write(individual.decode())

    proc = subprocess.Popen(['nvprof', '--csv', '-u', 'us',
                             './'+cudaAppName],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = proc.communicate(timeout=30) # second
        retcode = proc.poll()
        if retcode:
            print(stderr.decode())
            raise Exception('nvprof error')
    except subprocess.TimeoutExpired:
        # Offentime teminating nvprof will not teminate the underlying cuda program
        # So issue the kill command to those cuda app first
        print('8', end='', flush=True)
        subprocess.run(['killall', cudaAppName])
        proc.kill()
        proc.wait()
        stats['infinite'] = stats['infinite'] + 1
        return sys.float_info.max,

    program_output = stdout.decode()
    if program_output.find("Total Errors = 0") == -1:
        print('x', end='', flush=True)
        stats['invalid'] = stats['invalid'] + 1
        return sys.float_info.max,
    else:
        print('.', end='', flush=True)
        stats['valid'] = stats['valid'] + 1
        profile_output = stderr.decode()
        f = StringIO(profile_output)
        csv_list = list(csv.reader(f, delimiter=','))
        # search for kernel function
        for line in csv_list[5:]:
            # 7th column for name of CUDA function call
            if line[7].find(kernel) == 0:
                # third column for avg execution time
                return float(line[2]),

        raise Exception("{} is not a valid kernel function from nvprof".format(kernel))


def readLLVMsrc(str_encode):
    I = creator.Individual(str_encode)
    I.line_size = lineSize(I)
    I.cmd = []
    return I

def evole(llvm_src_filename: str, entry_kernel: str):
    stats = {'valid':0, 'invalid':0, 'infinite':0}

    try:
        f = open(llvm_src_filename, 'r')
        init_src_enc = f.read().encode()
    except IOError:
        print ("File {} does not exist".format(llvm_src_filename))
        exit()

    history = tools.History()
    toolbox = base.Toolbox()

    toolbox.register('evaluate', link_and_run, kernel=entry_kernel, stats=stats)
    toolbox.register('mutate', mutLLVM, kernel=entry_kernel, stats=stats)
    toolbox.register('mate', cxOnePointLLVM, init_src=init_src_enc)

    toolbox.register('select', tools.selTournament, tournsize=2)

    toolbox.register('individual', readLLVMsrc, init_src_enc)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Decorate the variation operators
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    pop = toolbox.population(n=100)
    popSize = len(pop)
    history.update(pop)
    # compile the initial llvm-IR into ptx and store it for later comparision
    translate_llvmIR_ptx(pop[0], "origin.ptx")

    pop[0].fitness.values = toolbox.evaluate(pop[0])
    for ind in pop:
        ind.fitness.values = pop[0].fitness.values

    fitnesses = [ ind.fitness.values for ind in pop ]
    fits = [ind.fitness.values[0] for ind in pop]

    generations = 0

    # Statistic
    maxFit = []
    avgFit = []
    minFit = []

    print("-- Generation 0 --")
    maxFit.append(max(fits))
    avgFit.append((sum(fits)/len(fits)))
    minFit.append(min(fits))
    print("  Max %s" % maxFit[-1])
    print("  Avg %s" % avgFit[-1])
    print("  Min %s" % minFit[-1])

    CXPB = 1.0
    MUPB = 0.9

    # while generations < 100:
    while True:
        generations = generations + 1
        count = 0
        # if generations < 4:
        #     MUPB = 1.0
        # else:
        #     MUPB = 0.2
        print("-- Generation %i --" % generations)

        offspring = toolbox.select(pop, popSize)
        # Preserve individual who has the highest fitness
        elite = tools.selBest(pop, 1)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        elite = list(map(toolbox.clone, elite))
        with open("best-{}.ll".format(generations), 'w') as f:
            f.write(elite[0].decode())

        # for child1, child2 in zip(offspring[::2], offspring[1::2]):
        #     if min(len(child1.cmd), len(child2.cmd)) < 2:
        #         continue
        #     if random.random() < CXPB and generations > 3:
        #         toolbox.mate(child1, child2)
        #         del child1.fitness.values
        #         del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUPB:
                print(count, end='', flush=True)
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [ toolbox.evaluate(ind) for ind in invalid_ind ]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring + elite
        # pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        maxFit.append(max(fits))
        avgFit.append((sum(fits)/len(fits)))
        minFit.append(min(fits))
        print("")
        print("  Max %s" % maxFit[-1])
        print("  Avg %s" % avgFit[-1])
        print("  Min %s" % minFit[-1])

    for fit, ind in zip(fits, pop):
        if fit == min(fits):
            translate_llvmIR_ptx(ind, 'final.ptx')

    print("valid variant:   {}".format(stats['valid']))
    print("invalid variant: {}".format(stats['invalid']))
    print("infinite variant:{}".format(stats['infinite']))

    # graph = nx.DiGraph(history.genealogy_tree)
    # graph = graph.reverse()     # Make the graph top-down
    # colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    # pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    # nx.draw(graph, pos, node_color=colors)
    # plt.savefig('genealogy.png')

if __name__ == '__main__':
    evole('matrixMul-cuda-nvptx64-nvidia-cuda-sm_35.ll', 'matrixMul_naive')
