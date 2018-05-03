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

log = open('debug_log', 'w')

def rearrage(cmd):
    c_cmd = [c for c in cmd if c[0] == '-c']
    r_cmd = [c for c in cmd if c[0] == '-r']
    i_cmd = [c for c in cmd if c[0] == '-i']
    s_cmd = [c for c in cmd if c[0] == '-s']

    cmd[:] = s_cmd + i_cmd + r_cmd + c_cmd
    return cmd

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
        fit = link_and_run(test_ind, kernel, stats)
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
        individual.line_size = lineSize(individual)
        individual.cmd.append(inst_UID)
        individual.fitness.values = fit
        return individual,

    print("Cannot get mutant to be compiled in {} trials".format(individual.line_size))
    return individual,

def cxOnePointLLVM(ind1, ind2, init_src, kernel, stats):
    # check whether they have the same starting edits(accessor)
    start_point = 0
    for edit1, edit2 in zip(ind1.cmd, ind2.cmd):
        if edit1 == edit2:
            start_point = start_point + 1
        else:
            break
    if (len(ind1.cmd)-1) <= start_point and (len(ind2.cmd)-1) <= start_point:
        print("s:{}, i:{}, i:{}. meaningless crossover".format(start_point, len(ind1.cmd)-1, len(ind2.cmd)-1),
              file=log)
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
    rearrage(cmd1)
    rearrage(cmd2)

    proc1 = subprocess.run(['llvm-mutate'] + [i for j in cmd1 for i in j],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           input=init_src)
    child1 = creator.Individual(proc1.stdout)
    print(cmd1, file=log, flush=True)
    print(proc1.stderr.decode(), file=log, flush=True)
    fit1 = [0]
    if proc1.returncode == 0:
        fit1 = link_and_run(child1, kernel, stats)
    if fit1[0] != 0:
        ind1[:] = bytearray(proc1.stdout)
        ind1.fitness.values = fit1
        ind1.line_size = lineSize(ind1)

    proc2 = subprocess.run(['llvm-mutate'] + [i for j in cmd2 for i in j],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           input=init_src)
    child2 = creator.Individual(proc2.stdout)
    print(cmd2, file=log, flush=True)
    print(proc2.stderr.decode(), file=log, flush=True)
    fit2 = [0]
    if proc2.returncode == 0:
        fit2 = link_and_run(child2, kernel, stats)
    if fit2[0] != 0:
        ind2[:] = bytearray(proc2.stdout)
        ind2.fitness.values = fit2
        ind2.line_size = lineSize(ind2)

    print('c', end='', flush=True)
    return ind1, ind2

def translate_llvmIR_ptx(llvmIR_str, outFileName="a.ptx"):
    proc = subprocess.run(['llc', '-o', outFileName],
                          stdout=subprocess.PIPE,
                          input=llvmIR_str)
    if proc.returncode is not 0:
        print(proc.stderr)
        raise Exception('llc error')

def link_and_run(individual, kernel, stats):
    cudaAppName = 'a.out'

    # link
    translate_llvmIR_ptx(individual)
    with open('a.ll', 'w') as f:
        f.write(individual.decode())

    # proc = subprocess.Popen(['nvprof', '--csv', '-u', 'us',
    proc = subprocess.Popen(['/usr/local/cuda/bin/nvprof',
                             '--unified-memory-profiling', 'off',
                             '--csv',
                             '-u', 'us',
                             './'+cudaAppName],
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
        subprocess.run(['killall', cudaAppName])
        proc.kill()
        proc.wait()
        stats['infinite'] = stats['infinite'] + 1
        return 0,

    program_output = stdout.decode()
    # if program_output.find("Total Errors = 0") == -1:
    if program_output.find("Test passed") == -1:
        print('x', end='', flush=True)
        stats['invalid'] = stats['invalid'] + 1
        return 0,
    else:
        print('.', end='', flush=True)
        stats['valid'] = stats['valid'] + 1
        profile_output = stderr.decode()
        f = StringIO(profile_output)
        csv_list = list(csv.reader(f, delimiter=','))

        # search for kernel function(s)
        kernel_time = []
        for line in csv_list[5:]:
            # 8th column for name of CUDA function call
            for name in kernel:
                if line[7].find(name) == 0:
                    # 3rd column for avg execution time
                    kernel_time.append(float(line[2]))

            if len(kernel) == len(kernel_time):
                return sum(kernel_time),

        raise Exception("{} is not a valid kernel function from nvprof".format(kernel))

def readLLVMsrc(str_encode):
    I = creator.Individual(str_encode)
    I.line_size = lineSize(I)
    I.cmd = []
    return I

def evolve(llvm_src_filename: str, entry_kernel, stats):
    for i in range(0, len(entry_kernel)):
        entry_kernel[i] = entry_kernel[i] + '('

    try:
        f = open(llvm_src_filename, 'r')
        init_src_enc = f.read().encode()
    except IOError:
        print("File {} does not exist".format(llvm_src_filename))
        exit()

    history = tools.History()
    toolbox = base.Toolbox()

    toolbox.register('evaluate', link_and_run, kernel=entry_kernel, stats=stats)
    toolbox.register('mutate', mutLLVM, kernel=entry_kernel, stats=stats)
    toolbox.register('mate', cxOnePointLLVM, init_src=init_src_enc, kernel=entry_kernel, stats=stats)

    toolbox.register('select', tools.selTournament, tournsize=2)

    toolbox.register('individual', readLLVMsrc, init_src_enc)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Decorate the variation operators
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    pop = toolbox.population(n=5)
    popSize = len(pop)
    history.update(pop)
    # compile the initial llvm-IR into ptx and store it for later comparision
    translate_llvmIR_ptx(pop[0], "origin.ptx")

    # Initial 3x mutate to get diverse population
    for ind in pop:
        toolbox.mutate(ind)
        toolbox.mutate(ind)
        toolbox.mutate(ind)

    pathlib.Path('stage').mkdir(exist_ok=True)
    fp = open("stage/initcmd.json", 'w')
    count = 0
    all_cmd = [ind.cmd for ind in pop]
    json.dump(all_cmd, fp, indent=2)
    for ind in pop:
        filename = "stage/" + str(count) + ".ll"
        with open(filename, 'w') as f:
            f.write(ind.decode())
        count = count + 1
    fp.close()

    fitnesses = [ind.fitness.values for ind in pop]
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

    CXPB = 0.8
    MUPB = 0.1

    # while generations < 100:
    while True:
        generations = generations + 1
        count = 0
        print("-- Generation %i --" % generations)

        offspring = toolbox.select(pop, popSize)
        # Preserve individual who has the highest fitness
        elite = tools.selBest(pop, 1)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        elite = list(map(toolbox.clone, elite))
        with open("best-{}.ll".format(generations-1), 'w') as f:
            f.write(elite[0].decode())

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if len(child1.cmd) < 2 and len(child2.cmd) < 2:
                continue
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

        for mutant in offspring:
            if random.random() < MUPB:
                count = count + 1
                print(count, end='', flush=True)
                del mutant.fitness.values
                toolbox.mutate(mutant)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind) for ind in invalid_ind]
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
    parser = argparse.ArgumentParser(description="Evolve CUDA kernel function")
    parser.add_argument('-k', '--kernel', help="Target kernel function. Use comma to separate kernels.")
    args = parser.parse_args()

    if args.kernel is None:
        print("Please specify the target kernel.",file=sys.stderr)
        parser.print_help()
        exit(1)

    kernel = args.kernel.split(',')

    stats = {'valid':0, 'invalid':0, 'infinite':0}
    try:
        evolve('cuda-device-only-kernel.ll', kernel, stats)
    except KeyboardInterrupt:
        print("valid variant:   {}".format(stats['valid']))
        print("invalid variant: {}".format(stats['invalid']))
        print("infinite variant:{}".format(stats['infinite']))
