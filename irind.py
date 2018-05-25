#!/usr/bin/env python3

import subprocess
import sys
import re
from collections import Counter

def llvmMutateWrap(srcEncIn, op:str, field1:str, field2:str):
    """
    return returnCode, mutated and encoded source, edit with UID
    """
    mut_command = ['llvm-mutate']
    if op == 'c':
        mut_command.extend(['-'+op, field1])
    else:
        mut_command.extend(['-'+op, field1 + ',' + field2])

    proc = subprocess.run(mut_command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          input=srcEncIn)
    if proc.returncode != 0:
        return -1, None, None
    if proc.stderr.decode().find('failed') != -1:
        return -2, srcEncIn, None
    if proc.stderr.decode().find('mismatch') != -1:
        return -3, srcEncIn, None

    mutateSrc = proc.stdout
    # read the uniqueID of the processed instructions
    for line in proc.stderr.decode().split('\n'):
        result = re.search('\w+ (U[0-9.irs]+)(,(U[0-9.irs]+))?', line)
        if result != None:
            break
    if result == None:
        print(proc.stderr.decode(), file=sys.stderr)
        with open('error.ll', 'w') as f:
            f.write(proc.stdout.decode())
        print(*mut_command)
        raise Exception("Could not understand the result from llvm-mutate")

    if op == 'c':
        editUID = ('-'+op, result.group(1))
    else:
        editUID = ('-'+op, result.group(1) + ',' + result.group(3))

    if proc.stderr.decode().find('no use') != -1:
        return 1, mutateSrc, editUID
    return 0, mutateSrc, editUID

def rearrage(cmd):
    # # this set approach reduces the duplicate edits in the list
    # cmdlist = list(set(cmd))
    # rearrage the edit sequence to reduce the fail chance of edit
    cmdlist = list(cmd)
    c_cmd = [c for c in cmdlist if c[0] == '-c']
    r_cmd = [c for c in cmdlist if c[0] == '-r']
    i_cmd = [c for c in cmdlist if c[0] == '-i']
    s_cmd = [c for c in cmdlist if c[0] == '-s']

    cmdlist = s_cmd + i_cmd + r_cmd + c_cmd
    return cmdlist

def diff(edits1, edits2):
    # sharedEdits = set(edits1).intersection(edits2)
    # diff1 = set(edits1) - sharedEdits
    # diff2 = set(edits2) - sharedEdits
    # return list(sharedEdits), list(diff1), list(diff2)
    c1 = Counter(edits1)
    c2 = Counter(edits2)
    diff1 = c1 - c2
    diff2 = c2 - c1
    sharedEdits = c1 - diff1
    return list(sharedEdits.elements()), list(diff1.elements()), list(diff2.elements())

class llvmIRrep:
    edits = []
    srcEnc = ""
    lineSize = 0

    def __init__(self, srcEnc, edits=None):
        self.srcEnc = srcEnc
        if edits is None:
            self.edits = []
        else:
            self.edits = edits
        self.update_linesize()

    def update_linesize(self):
        try:
            readline_proc = subprocess.run(['llvm-mutate', '-I'],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           input=self.srcEnc,
                                           check=True )
        except subprocess.CalledProcessError as err:
            print(err.stderr, file=sys.stderr)
            raise Exception('llvm-mutate error in calculating line size')

        self.lineSize = int(readline_proc.stderr.decode())

    def update(self, srcEnc):
        self.srcEnc = srcEnc
        self.update_linesize()

    def update_from_edits(self):
        proc = subprocess.run(['llvm-mutate'] + [arg for edit in self.edits for arg in edit],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              input=self.srcEnc)
        if proc.returncode != 0:
            # print(proc.stderr.decode())
            return False

        self.update(proc.stdout)
        return True

    def rearrage(self):
        # self.edits = list(set(self.edits))
        c_cmd = [c for c in self.edits if c[0] == '-c']
        r_cmd = [c for c in self.edits if c[0] == '-r']
        i_cmd = [c for c in self.edits if c[0] == '-i']
        s_cmd = [c for c in self.edits if c[0] == '-s']

        self.edits = s_cmd + i_cmd + r_cmd + c_cmd

    def ptx(self, outf):
        proc = subprocess.run(['llc', '-o', outf],
                              stdout=subprocess.PIPE,
                              input=self.srcEnc)

        if proc.returncode is not 0:
            print(proc.stderr)
            raise Exception('llc error')