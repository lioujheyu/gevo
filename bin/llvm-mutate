#!/usr/bin/env python3
import argparse
import sys
import subprocess
import inspect
import os

import gevo
from gevo._llvm import __llvm_version__

os.path.dirname(inspect.getfile(gevo))
LLVM_MUTATE_LIBRARY_PATH=f'{os.path.dirname(inspect.getfile(gevo))}/Mutate.so.{__llvm_version__}'

if __name__ == '__main__':
    # Command parser
    class MutationAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if 'mutation_commands' not in namespace:
                setattr(namespace, 'mutation_commands', [])
            previous_act = namespace.mutation_commands
            previous_act.append((self.dest, values))
            setattr(namespace, 'mutation_commands', previous_act)

    llvm_mutate_parser = argparse.ArgumentParser(description="Manipulate LLVM assembly from Nvidia CUDA Kernels")
    llvm_mutate_parser.add_argument('-o', '--output_file', metavar='FILE', nargs='?', type=argparse.FileType('w'), default=sys.stdout,
        help='Output file name. Output will be redirected to stdout if output file name is not specified')
    llvm_mutate_parser.add_argument('-f', '--input_file', metavar='FILE', nargs='?', type=argparse.FileType('rb'), default=sys.stdin,
        help='Input file name. llvm-mutate will use and wait for stdin as the input stream if input file name is not specified.')
    llvm_mutate_parser.add_argument('-n', '--name', action='store_true',
        help='Add unique ID (UID) for each instruction. UID is one of needed instruction description in mutation operations')
    llvm_mutate_parser.add_argument('-I', '--ids', action='store_true',
        help='Show the number of instructions')
    llvm_mutate_parser.add_argument('--not_use_result', action='store_true',
        help='Not connect the newly inserted instruction\'s result value back into the use-def\
              chain when performing mutation operations. This argument is mainly for reproducing\
              program variant from a sequence of mutations')

    # grouping mutation commands
    mutation_operation_group = llvm_mutate_parser.add_argument_group(
        title='Mutation Operations',
        description='Mutation operations only accept instruction description [INST] in 2 formats: \
                     integer number as instruction index or Unique ID. \
                     Note: --name, --ids, and mutation operation cannot be used together')
    mutation_operation_group.add_argument(
        '-c', '--cut', type=str, dest='-cut', action=MutationAction, metavar='INST',
        help='Cut the given instruction')
    mutation_operation_group.add_argument(
        '-i', '--insert', type=str, dest='-insert', action=MutationAction, metavar='INST1,INST2',
        help='Copy the second inst. before the first')
    mutation_operation_group.add_argument(
        '-p', '--oprepl', type=str, dest='-oprepl', action=MutationAction, metavar='INST1,INST2',
        help='Replace the first Operand. with the second')
    mutation_operation_group.add_argument(
        '-r', '--replace', type=str, dest='-replace', action=MutationAction, metavar='INST1,INST2', 
        help='Replace the first inst. with the second')
    mutation_operation_group.add_argument(
        '-m', '--move', type=str, dest='-move', action=MutationAction, metavar='INST1,INST2',
        help='Move the second inst. before the first')
    mutation_operation_group.add_argument(
        '-s', '--swap', type=str, dest='-swap', action=MutationAction, metavar='INST1,INST2',
        help='Swap the location of two instructions')

    args = llvm_mutate_parser.parse_args()
    OPT_NOT_USE_RESULT = '-use_result=0' if args.not_use_result else ''

    if args.name and 'mutation_commands' in args:
        print("--name and mutation operations cannot be used together.")
        sys.exit(-1)

    if args.name:
        opt_proc = subprocess.Popen(
            [ f'opt-{__llvm_version__}',
              '-load', LLVM_MUTATE_LIBRARY_PATH,
              '-name'],
            stdin=args.input_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE )
        opt_stdout, opt_stderr = opt_proc.communicate()
        if opt_proc.returncode != 0:
            print(f"llvm-mutate: Error in {opt_proc.args}")
            print(opt_stderr.decode(), end='')
            sys.exit(-1)
    elif args.ids:
        opt_proc = subprocess.Popen(
            [ f'opt-{__llvm_version__}',
              '-load', LLVM_MUTATE_LIBRARY_PATH,
              '-ids'],
            stdin=args.input_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE )
        opt_stdout, opt_stderr = opt_proc.communicate()
        if opt_proc.returncode != 0:
            print(f"llvm-mutate: Error in {opt_proc.args}")
            print(opt_stderr.decode(), end='')
            sys.exit(-1)
        print(opt_stderr.decode(), end='', file=sys.stderr)
        sys.exit(0)
    elif 'mutation_commands' in args:
        input_str = args.input_file.buffer.read()
        for mop in args.mutation_commands:
            insts = mop[1].split(',')
            inst_args = [ f'-inst1={insts[0]}' ] if len(insts) == 1 else\
                        [ f'-inst1={insts[0]}', f'-inst2={insts[1]}']
            opt_args = [f'opt-{__llvm_version__}',
                        '-load', LLVM_MUTATE_LIBRARY_PATH,
                        '-use_result=0', mop[0]] if args.not_use_result else\
                       [f'opt-{__llvm_version__}',
                        '-load', LLVM_MUTATE_LIBRARY_PATH, mop[0]]

            opt_args.extend(inst_args)

            opt_proc = subprocess.Popen(
                opt_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE )
            opt_stdout, opt_stderr = opt_proc.communicate(input=input_str)

            if opt_proc.returncode != 0:
                print(f"llvm-mutate: opt error in {mop[0]} {mop[1]}")
                print(opt_stderr.decode(), end='')
                sys.exit(-1)

            # TODO: Have proper return code from Mutate.so. irind.py in gevo need to be changed as well
            print(opt_stderr.decode(), end='', file=sys.stderr)
            if opt_stderr.decode().find('failed') != -1:
                sys.exit(0)
            if opt_stderr.decode().find('mismatch') != -1:
                sys.exit(0)
            if opt_stderr.decode().find('no use') != -1:
                sys.exit(0)
            input_str = opt_stdout
    else:
        llvm_mutate_parser.print_help()
        sys.exit(-1)

    llvmdis_proc = subprocess.Popen(
        [ f'llvm-dis-{__llvm_version__}'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE )
    llvmdis_stdout, llvmdis_stderr = llvmdis_proc.communicate(input=opt_stdout)
    if llvmdis_proc.returncode != 0:
        print(f"llvm-mutate: llc error in \"{' '.join(opt_proc.args)} | {' '.join(llvmdis_proc.args)}\"")
        print(llvmdis_stderr.decode())
        sys.exit(-1)

    print(llvmdis_stdout.decode(), file=args.output_file, end='')

    # Link to PTX
    # cuda.init()
    # SM_MAJOR, SM_MINOR = cuda.Device(0).compute_capability()
    # MGPU = 'sm_' + str(SM_MAJOR) + str(SM_MINOR)

    # llc_proc = subprocess.Popen(
    #     [ f'llc{LLVM_VERSION}', "-march=nvptx64", "-mcpu="+MGPU, "-mattr=+ptx70"],
    #     # [ f'llc{LLVM_VERSION}'],
    #     stdin=subprocess.PIPE,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE )
    # llc_stdout, llc_stderr = llc_proc.communicate(input=opt_stdout)
    # if llc_proc.returncode != 0:
    #     print(f"llvm-mutate: llc error in \"{' '.join(opt_proc.args)} | {' '.join(llc_proc.args)}\"")
    #     print(llc_stderr.decode())
    #     sys.exit(-1)
