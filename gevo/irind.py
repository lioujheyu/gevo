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
    if op == 'c' or op == 'x':
        mut_command.extend(['-'+op, field1])
    else:
        mut_command.extend(['-'+op, field1 + ',' + field2])

    proc = subprocess.run(mut_command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          input=srcEncIn)

    if proc.returncode != 0:
        print(proc.stderr.decode(), file=sys.stderr)
        with open('error.ll', 'w') as f:
            f.write(proc.stdout.decode())
        print(*mut_command)
        return -1, srcEncIn, None
    if proc.stderr.decode().find('failed') != -1:
        return -2, srcEncIn, None
    if proc.stderr.decode().find('mismatch') != -1:
        return -3, srcEncIn, None
    if proc.stderr.decode().find('no use') != -1:
        return -4, srcEncIn, None

    mutateSrc = proc.stdout
    # read the uniqueID of the processed instructions
    editUID = []
    for line in proc.stderr.decode().split('\n'):
        if len(line) == 0:
            continue
        result = re.search('(\w+) (U[0-9.irsmxOP]+)(,([UAC][0-9.irsmx]+))?', line)

        if result is None:
            print(proc.stderr.decode(), file=sys.stderr)
            with open('error.ll', 'w') as f:
                f.write(proc.stdout.decode())
            print(*mut_command)
            raise Exception("Could not understand the result from llvm-mutate")

        try:
            if result.group(1) == "opreplaced":
                editUID.append(('-p', result.group(2) + ',' + result.group(4)))
            else:
                if op == 'c' or op == 'x':
                    editUID = [('-'+op, result.group(2))] + editUID
                else:
                    editUID = [('-'+op, result.group(2) + ',' + result.group(4))] + editUID
        except TypeError:
            print(proc.stderr.decode(), file=sys.stderr)
            with open('error.ll', 'w') as f:
                f.write(proc.stdout.decode())
            print(*mut_command)
            raise Exception("Could not understand the result from llvm-mutate")

    if proc.stderr.decode().find('no use') != -1:
        return 1, mutateSrc, editUID
    return 0, mutateSrc, editUID

def rearrage(cmd):
    cmdlist = list(cmd)
    c_cmd  = sorted([c for c in cmdlist if c[0][0] == '-c'])
    r_cmd  = sorted([c for c in cmdlist if c[0][0] == '-r'])
    i_cmd  = sorted([c for c in cmdlist if c[0][0] == '-i'])
    m_cmd  = sorted([c for c in cmdlist if c[0][0] == '-m'])
    s_cmd  = sorted([c for c in cmdlist if c[0][0] == '-s'])
    x_cmd  = sorted([c for c in cmdlist if c[0][0] == '-x'])
    op_cmd = sorted([c for c in cmdlist if c[0][0] == '-p'])

    cmdlist = s_cmd + m_cmd + i_cmd + r_cmd + c_cmd + x_cmd + op_cmd
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
        ['llvm-mutate', '--not_use_result'] + ind.serialize_edits(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=ind.srcEnc
    )
    if proc.returncode != 0:
        resultList[idx] = False
    else:
        ind.update(proc.stdout)
        resultList[idx] = True

def serialize_edits_to_str(edits):
    outstr = "+".join([";".join(["{} {}".format(edit[0], edit[1]) for edit in editG]) for editG in edits ])
    return outstr

def edits_as_key(edits):
    return tuple([edit for editG in edits for edit in editG])

class llvmIRrep():
    def __init__(self, srcEnc, mgpu, edits=None, mattr="+ptx70"):
        # default compilation argument to Nvidia pascal architecture.
        self.mgpu = mgpu
        self.mattr = mattr
        self.srcEnc = srcEnc
        if edits is None:
            self.edits = []
        else:
            self.edits = edits
        self.update_linesize()

    def key(self):
        return tuple([edit for editG in self.edits for edit in editG])

    def __len__(self):
        return len(self.edits)

    def __eq__(self, other):
        return self.edits == other.edits

    def __hash__(self):
        return hash(self.key())

    def serialize_edits(self):
        if self.edits is None:
            return None
        return [arg for editG in self.edits for edit in editG for arg in edit]

    def serialize_edits_to_str(self):
        outstr = "+".join([";".join(["{} {}".format(edit[0], edit[1]) for edit in editG]) for editG in self.edits ])
        return outstr


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
        if sweepEdits is False:
            proc = subprocess.run(['llvm-mutate', '--not_use_result'] + self.serialize_edits(),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  input=self.srcEnc)
            if proc.returncode != 0:
                return False
            mutateSrcEn = proc.stdout
        else:
            raise Exception('Not fixed yet')
            # validEdits = []
            # mutateSrcEn = self.srcEnc
            # for edit in self.edits:
            #     rc, mutateSrcEn, _ = llvmMutateWrap(
            #                             mutateSrcEn,
            #                             edit[0][1],
            #                             edit[1].split(',')[0],
            #                             edit[1].split(',')[1] if ',' in edit[1] else None)
            #     if rc < 0:
            #         continue
            #     validEdits.append(edit)

            # self.edits = validEdits

        self.update(mutateSrcEn)
        return True

    def rearrage(self):
        c_cmd  = sorted([c for c in self.edits if c[0][0] == '-c'])
        r_cmd  = sorted([c for c in self.edits if c[0][0] == '-r'])
        i_cmd  = sorted([c for c in self.edits if c[0][0] == '-i'])
        m_cmd  = sorted([c for c in self.edits if c[0][0] == '-m'])
        s_cmd  = sorted([c for c in self.edits if c[0][0] == '-s'])
        x_cmd  = sorted([c for c in self.edits if c[0][0] == '-x'])
        op_cmd = sorted([c for c in self.edits if c[0][0] == '-p'])

        self.edits = s_cmd + m_cmd + i_cmd + r_cmd + c_cmd + x_cmd + op_cmd

    def ptx(self, outf):
        proc = subprocess.run(['llc', "-march=nvptx64", "-mcpu="+self.mgpu, "-mattr="+self.mattr, '-o', outf],
                              stdout=subprocess.PIPE,
                              input=self.srcEnc)

        if proc.returncode != 0:
            print(proc.stderr)
            print(self.edits)
            raise Exception('llc error')
