#!/usr/bin/env python3

import subprocess
import sys

def rearrage(cmd):
    # this set approach reduces the duplicate edits in the list
    cmd[:] = list(set(cmd))
    # rearrage the edit sequence to reduce the fail chance of edit
    c_cmd = [c for c in cmd if c[0] == '-c']
    r_cmd = [c for c in cmd if c[0] == '-r']
    i_cmd = [c for c in cmd if c[0] == '-i']
    s_cmd = [c for c in cmd if c[0] == '-s']

    cmd = s_cmd + i_cmd + r_cmd + c_cmd
    return cmd

def diff(edits1, edits2):
    sharedEdits = set(edits1).intersection(edits2)
    diff1 = set(edits1) - sharedEdits
    diff2 = set(edits2) - sharedEdits
    return list(sharedEdits), list(diff1), list(diff2)

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
            print(proc.stderr.decode())
            return False

        self.update(proc.stdout)
        return True

    def rearrage(self):
        self.edits[:] = list(set(self.edits))
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