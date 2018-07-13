#!/usr/bin/env python3

import argparse
import subprocess
import random
import csv
import json
import pathlib
import sys
import filecmp
import io
from threading import Thread
from threading import Lock

import matplotlib.pyplot as plt
# import networkx as nx
import numpy
from deap import base
from deap import creator
from deap import tools
import pptx
from PIL import Image

sys.path.append('/home/jliou4/genetic-programming/cuda_evolve')
import irind
from irind import llvmMutateWrap
from irind import update_from_edits
import fuzzycompare

# critical section of multithreading
lock = Lock()

class evolution:
    # Parameters
    log = open('debug_log', 'w')
    cudaPTX = 'a.ptx'

    # Content
    pop = []
    generation = 0
    presentation = pptx.Presentation()

    mutStats = {
        'valid':0, 'invalid':0, 'infinite':0,
        'maxFit':[], 'avgFit':[], 'minFit':[]
    }
    def __init__(self, kernel, bin, args="", timeout=30, fitness='time',
                 llvm_src_filename='cuda-device-only-kernel.ll',
                 compare_filename="compare.json",
                 CXPB=0.8, MUPB=0.1):
        self.CXPB = CXPB
        self.MUPB = MUPB
        self.kernels = kernel
        self.appBinary = bin
        self.appArgs = "" if args is None else args
        self.timeout = timeout
        self.fitness_function = fitness

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

        # tools initialization
        # Minimize both performance and error
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", irind.llvmIRrep, fitness=creator.FitnessMin)
        self.history = tools.History()
        self.toolbox = base.Toolbox()
        self.toolbox.register('mutate', self.mutLLVM)
        self.toolbox.register('mate', self.cxOnePointLLVM)
        # self.toolbox.register('select', tools.selDoubleTournament, fitness_size=2, parsimony_size=1.4, fitness_first=True)
        self.toolbox.register('select', tools.selNSGA2)
        self.toolbox.register('individual', creator.Individual, srcEnc=self.initSrcEnc)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        # Decorate the variation operators
        self.toolbox.decorate("mate", self.history.decorator)
        self.toolbox.decorate("mutate", self.history.decorator)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("min", min)
        self.stats.register("max", max)

        self.logbook = tools.Logbook()
        self.paretof = tools.ParetoFront()
        self.logbook.header = "gen", "evals", "min", "max"

    # def printGen(self, gen):
    #     print("-- Generation %s --" % gen)
    #     print("  Max {}".format(self.stats['maxFit'][gen]))
    #     # print("  Avg %s" % self.stats['avgFit'][gen])
    #     print("  Min {}".format(self.stats['minFit'][gen]))

    def updateSlideFromPlot(self):
        pffits = [ind.fitness.values for ind in self.paretof]
        fits = [ind.fitness.values for ind in self.pop if ind not in pffits]
        plt.title("Program variant performance - Generation {}".format(self.generation))
        plt.xlabel("Runtime")
        plt.ylabel("Error")
        plt.scatter([fit[0] for fit in fits], [fit[1] for fit in fits],
                    marker='*', label="dominated")
        plt.scatter([pffit[0] for pffit in pffits], [pffit[1] for pffit in pffits],
                    marker='o', c='red', label="pareto front")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)

        slide = self.presentation.slides.add_slide(self.presentation.slide_layouts[6])
        # left = (self.presentation.slide_width - pptx.util.px(640))/2
        # top = (self.presentation.slide_height - pptx.util.px(480))/2
        left = top = pptx.util.Inches(1)
        pic = slide.shapes.add_picture(img, left, top)

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

        err = 0.0
        if self.verifier['mode'] == 'string':
            result = False if src.find(golden) == -1 else True
            return result, err
        elif self.verifier['mode'] == 'file':
            result = True
            for s, g in zip(src, golden):
                if self.verifier.get('fuzzy', False) == False:
                    try:
                        result = result & filecmp.cmp(s, g)
                    except IOError:
                        print("File {} or {} cannot be found".format(src, golden))
                else:
                    rc, msg, maxerr, avgerr = fuzzycompare.file(s, g)
                    if rc < 0:
                        raise Exception(msg)
                    result = result & (True if rc==0 else False)
                    err = maxerr if maxerr > err else err
            return result, err
        else:
            raise Exception("Unknown comparing mode \"{}\" from compare.json".format(
                self.verifier['mode']))

    def mutLLVM(self, individual):
        trial = 0
        # cut, replace, insert, swap
        operations = ['c', 'r', 'i', 's', 'm']
        while trial < individual.lineSize:
            trial = trial + 1
            line1 = random.randint(1, individual.lineSize)
            line2 = random.randint(1, individual.lineSize)
            while line1 == line2:
                line2 = random.randint(1, individual.lineSize)

            op = random.choice(operations)
            rc, mutateSrc, editUID = llvmMutateWrap(individual.srcEnc, op, str(line1), str(line2))
            if rc < 0:
                continue

            test_ind = creator.Individual(self.initSrcEnc)
            test_ind.edits[:] = individual.edits + [editUID]
            test_ind.rearrage()
            if test_ind.update_from_edits() == False:
                continue

            with lock:
                fit = self.evaluate(test_ind)
                print('m', end='', flush=True)
            if None in fit:
                continue

            individual.update(srcEnc=test_ind.srcEnc)
            individual.edits.append(editUID)
            individual.rearrage()
            individual.fitness.values = fit
            return individual,

        print("Cannot get mutant to be compiled in {} trials".format(individual.lineSize))
        return individual,

    def cxOnePointLLVM(self, ind1, ind2):
        shuffleEdits = ind1.edits + ind2.edits
        random.shuffle(shuffleEdits)
        point = random.randint(1, len(shuffleEdits)-1)
        cmd1 = shuffleEdits[:point]
        cmd2 = shuffleEdits[point:]
        cmd1 = irind.rearrage(cmd1)
        cmd2 = irind.rearrage(cmd2)

        child1 = creator.Individual(self.initSrcEnc)
        child1.edits = list(cmd1)
        child1.update_from_edits(sweepEdits=False)
        child2 = creator.Individual(self.initSrcEnc)
        child2.edits = list(cmd2)
        child2.update_from_edits(sweepEdits=False)

        with lock:
            fit1 = self.evaluate(child1)
            fit2 = self.evaluate(child2)
            print('c', end='', flush=True)

        if None in fit1:
            ind1 = child1
        if None in fit2:
            ind2 = child2

        return ind1, ind2

    def evaluate(self, individual):
        # link
        individual.ptx(self.cudaPTX)
        with open('a.ll', 'w') as f:
            f.write(individual.srcEnc.decode())

        proc = subprocess.Popen(['/usr/local/cuda/bin/nvprof',
                                 '--unified-memory-profiling', 'off',
                                 '--profile-from-start', 'off',
                                 '--system-profiling', 'on',
                                 '--csv',
                                 '-u', 'us',
                                 './' + self.appBinary] + self.appArgs,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout) # second
            retcode = proc.poll()
            # retcode == 9: error is from testing program, not nvprof
            # retcode == 15: Target program receive segmentation fault
            if retcode == 9 or retcode == 15:
                print('x', end='', flush=True)
                self.mutStats['invalid'] = self.mutStats['invalid'] + 1
                return None, None
            # Unknown nvprof error
            if retcode != 0:
                print(stderr.decode(), file=sys.stderr)
                raise Exception('nvprof error')
        except subprocess.TimeoutExpired:
            # Sometimes terminating nvprof will not terminate the underlying cuda program
            # if that program is corrupted. So issue the kill command to those cuda app first
            print('8', end='', flush=True)
            subprocess.run(['killall', self.appBinary],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
            proc.kill()
            proc.wait()
            self.mutStats['infinite'] = self.mutStats['infinite'] + 1
            return None, None

        program_output = stdout.decode()
        cmpResult, err = self.resultCompare(program_output)
        if cmpResult is False:
            print('x', end='', flush=True)
            self.mutStats['invalid'] = self.mutStats['invalid'] + 1
            return None, None
        else:
            print('.', end='', flush=True)
            self.mutStats['valid'] = self.mutStats['valid'] + 1
            profile_output = stderr.decode()
            csv_list = list(csv.reader(io.StringIO(profile_output), delimiter=','))

            # search for kernel function(s)
            kernel_time = []
            energy = None
            # The stats starts after 5th line
            for line in csv_list[5:]:
                if len(line) == 0:
                    continue
                if line[0] == "GPU activities":
                    for name in self.kernels:
                        # 8th column for name of CUDA function call
                        try:
                            if line[7].split('(')[0] == name:
                                # 3rd column for avg execution time
                                kernel_time.append(float(line[2]))
                        except:
                            continue
                if line[0] == "Power (mW)":
                    count = int(line[2])
                    avg_power = float(line[3])
                    # The emprical shows that the sampling frequency is around 50Hz
                    energy = count * avg_power / 20

            if len(self.kernels) == len(kernel_time) and energy is not None:
                if self.fitness_function == 'time':
                    return sum(kernel_time), err
                elif self.fitness_function == 'power':
                    return energy, err
            else:
                print("Can not find kernel \"{}\" from nvprof".format(self.kernels), file=sys.stderr)
                return None, None

    def evolve(self, resumeGen):
        threadPool = []
        if resumeGen == -1:
            popSize = 100
            print("Initialize the population. Size {}".format(popSize))
            self.pop = self.toolbox.population(n=popSize)

            # Initial 3x mutate to get diverse population
            for ind in self.pop:
                self.toolbox.mutate(ind)
                self.toolbox.mutate(ind)
                self.toolbox.mutate(ind)
            self.writeStage()
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

            # popSize = len(allEdits)
            popSize = 1
            print("Resume the population from {}. Size {}".format(stageFileName, popSize))
            self.pop = self.toolbox.population(n=popSize)
            self.generation = resumeGen

            resultList = [False] * popSize
            for i, (edits, ind) in enumerate(zip(allEdits, self.pop)):
                editsList = [(e[0], e[1]) for e in edits]
                ind.edits = editsList
                threadPool.append(
                    Thread(target=update_from_edits, args=(i, ind, resultList))
                )
                threadPool[-1].start()

            for i, ind in enumerate(self.pop):
                threadPool[i].join()
                if resultList[i] == False:
                    raise Exception("Could not reconstruct ind from edits:{}".format(ind.edits))
                fitness = self.evaluate(ind)
                if None in fitness:
                    for edit in ind.edits:
                        print(edit)
                    raise Exception("Encounter invalid individual during reconstruction")
                ind.fitness.values = fitness

        # This is to assign the crowding distance to the individuals
        # and also to sort the pop with front rank
        self.pop = self.toolbox.select(self.pop, popSize)
        self.history.update(self.pop)
        record = self.stats.compile(self.pop)
        self.paretof.update(self.pop)
        self.logbook.record(gen=0, evals=popSize, **record)
        print("")
        print(self.logbook.stream)
        self.updateSlideFromPlot()

        # pffits = [ind.fitness.values for ind in self.paretof]
        # fits = [ind.fitness.values for ind in self.pop if ind not in pffits]
        # plt.scatter([fit[0] for fit in fits], [fit[1] for fit in fits], marker='*')
        # plt.scatter([pffits[0] for fit in fits], [pffits[1] for fit in fits], marker='o', c=red)
        # plt.savefig(str(self.generation) + '.png')

        while True:
            offspring = tools.selTournamentDCD(self.pop, popSize)
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            with open("best-{}.ll".format(self.generation), 'w') as f:
                f.write(self.pop[0].srcEnc.decode())
            with open("best-{}.edit".format(self.generation), 'w') as f:
                print(self.pop[0].edits, file=f)

            self.generation = self.generation + 1

            threadPool.clear()
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if len(child1) < 2 and len(child2) < 2:
                    continue
                if random.random() < self.CXPB:
                    threadPool.append(
                        Thread(target=self.toolbox.mate, args=(child1, child2)))
                    threadPool[-1].start()
            for thread in threadPool:
                thread.join()

            threadPool.clear()
            for mutant in offspring:
                if random.random() < self.MUPB:
                    del mutant.fitness.values
                    threadPool.append(
                        Thread(target=self.toolbox.mutate, args=(mutant,)))
                    threadPool[-1].start()
            for thread in threadPool:
                thread.join()


            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [self.evaluate(ind) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            self.pop = self.toolbox.select(self.pop + offspring, popSize)
            record = self.stats.compile(self.pop)
            self.logbook.record(gen=self.generation, evals=popSize, **record)
            self.paretof.update(self.pop)

            print("")
            print(self.logbook.stream)
            self.updateSlideFromPlot()
            self.writeStage()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evolve CUDA kernel function")
    parser.add_argument('-k', '--kernel', type=str, required=True,
        help="Target kernel function of the given CUDA application. Use comma to separate kernels.")
    parser.add_argument('-r', '--resume', type=int, default=-1,
        help="Resume the process from genetating the population by reading stage/<RESUME>.json")
    parser.add_argument('-t', '--timeout', type=int, default=30,
        help="The timeout period to evaluate the CUDA application")
    parser.add_argument('-fitf', '--fitness_function', type=str, default='time',
        help="What is the target fitness for the evolution. Default ot execution time. Can be changed to power")
    parser.add_argument('binary',help="Binary of the CUDA application", nargs='?', default='a.out')
    parser.add_argument('args',help="arguments for the application binary", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    kernel = args.kernel.split(',')
    evo = evolution(
        kernel=kernel,
        bin=args.binary,
        args=args.args,
        timeout=args.timeout,
        fitness=args.fitness_function)

    print("      Target CUDA program: {}".format(args.binary))
    print("Args for the CUDA program: {}".format(" ".join(args.args)))
    print("           Target kernels: {}".format(" ".join(kernel)))
    print("       Evaluation Timeout: {}".format(args.timeout))
    print("         Fitness function: {}".format(args.fitness_function))

    try:
        evo.evolve(args.resume)
    except KeyboardInterrupt:
        subprocess.run(['killall', args.binary])
        print("valid variant:   {}".format(evo.stats['valid']))
        print("invalid variant: {}".format(evo.stats['invalid']))
        print("infinite variant:{}".format(evo.stats['infinite']))
        if evo.generation > 0:
            evo.presentation.save('progress.pptx')
