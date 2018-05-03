#!/usr/bin/env python3

import subprocess
import sys

class llvmIRrep:
    edits = [()]
    srcEnc = ""
    lineSize = 0

    def __init__(self, srcEnc, edits=None):
        self.srcEnc = srcEnc
        if edits is not None:
            self.edits = edits

    def update_linesize(self):
        try:
            readline_proc = subprocess.run( ['llvm-mutate', '-I'],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            input=self.srcEnc,
                                            check=True )
        except subprocess.CalledProcessError as err:
            print(err.stderr, file=sys.stderr)
            raise Exception('llvm-mutate error in calculating line size')

        self.lineSize = int(readline_proc.stderr.decode())

    def rearrage(self):
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