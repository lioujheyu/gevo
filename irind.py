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
        return -1, srcEncIn, None
    if proc.stderr.decode().find('failed') != -1:
        return -2, srcEncIn, None
    if proc.stderr.decode().find('mismatch') != -1:
        return -3, srcEncIn, None
    if proc.stderr.decode().find('no use') != -1:
        return -4, srcEncIn, None

    mutateSrc = proc.stdout
    # read the uniqueID of the processed instructions
    for line in proc.stderr.decode().split('\n'):
        result = re.search('\w+ (U[0-9.irsm]+)(,(U[0-9.irsm]+))?', line)
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
    cmdlist = list(cmd)
    c_cmd = [c for c in cmdlist if c[0] == '-c']
    r_cmd = [c for c in cmdlist if c[0] == '-r']
    i_cmd = [c for c in cmdlist if c[0] == '-i']
    m_cmd = [c for c in cmdlist if c[0] == '-m']
    s_cmd = [c for c in cmdlist if c[0] == '-s']

    cmdlist = s_cmd + m_cmd + i_cmd + r_cmd + c_cmd
    return cmdlist

def diff(edits1, edits2):
    c1 = Counter(edits1)
    c2 = Counter(edits2)
    diff1 = c1 - c2
    diff2 = c2 - c1
    sharedEdits = c1 - diff1
    return list(sharedEdits.elements()), list(diff1.elements()), list(diff2.elements())

def update_from_edits(idx, ind, resultList):
    proc = subprocess.run(
        ['llvm-mutate'] + [arg for edit in ind.edits for arg in edit],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=ind.srcEnc
    )
    if proc.returncode != 0:
        resultList[idx] = False

    ind.update(proc.stdout)
    resultList[idx] = True

class llvmIRrep():
    def __init__(self, srcEnc, edits=None):
        self.srcEnc = srcEnc
        if edits is None:
            self.edits = []
        else:
            self.edits = edits
        self.update_linesize()

    def __len__(self):
        return len(self.edits)

    def update_linesize(self):
        try:
            readline_proc = subprocess.run(['llvm-mutate', '-I'],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           input=self.srcEnc,
                                           check=True)
        except subprocess.CalledProcessError as err:
            print(err.stderr, file=sys.stderr)
            raise Exception('llvm-mutate error in calculating line size')

        self.lineSize = int(readline_proc.stderr.decode())

    def update(self, srcEnc):
        self.srcEnc = srcEnc
        self.update_linesize()

    def update_from_edits(self, sweepEdits=False):
        if sweepEdits == False:
            proc = subprocess.run(['llvm-mutate'] + [arg for edit in self.edits for arg in edit],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  input=self.srcEnc)
            if proc.returncode != 0:
                return False
            mutateSrcEn = proc.stdout
        else:
            validEdits = []
            mutateSrcEn = self.srcEnc
            for edit in self.edits:
                rc, mutateSrcEn, _ = llvmMutateWrap(
                                        mutateSrcEn,
                                        edit[0][1],
                                        edit[1].split(',')[0],
                                        edit[1].split(',')[1] if ',' in edit[1] else None)
                if rc < 0:
                    continue
                validEdits.append(edit)

            self.edits = validEdits

        self.update(mutateSrcEn)
        return True

    def rearrage(self):
        c_cmd = [c for c in self.edits if c[0] == '-c']
        r_cmd = [c for c in self.edits if c[0] == '-r']
        i_cmd = [c for c in self.edits if c[0] == '-i']
        m_cmd = [c for c in self.edits if c[0] == '-m']
        s_cmd = [c for c in self.edits if c[0] == '-s']

        self.edits = s_cmd + m_cmd + i_cmd + r_cmd + c_cmd

    def ptx(self, outf):
        proc = subprocess.run(['llc', '-o', outf],
                              stdout=subprocess.PIPE,
                              input=self.srcEnc)

        if proc.returncode is not 0:
            print(proc.stderr)
            print(self.edits)
            raise Exception('llc error')
