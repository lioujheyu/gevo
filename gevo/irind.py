#!/usr/bin/env python3

import subprocess
import sys
import re

try:
    from gevo._llvm import __llvm_version__
except ImportError:
    pass # A testing path

class edit(tuple):
    def __new__(cls, iterable):
        return super().__new__(cls, iterable)

    def serialize(self):
        return [UID for e in self for UID in e]

def encode_edits_from_list(lists):
    '''For reading the edits from a json file where tuple is decode into list'''
    sublist = []
    for ele in lists:
        if isinstance(ele[0], list) or isinstance(ele[0], tuple):
            sublist.append(encode_edits_from_list(ele))
        elif isinstance(ele[0], str):
            sublist = sublist + [tuple(ele)]

    if isinstance(lists[0][0], list) or isinstance(lists[0][0], tuple):
        return tuple(sublist)
    elif isinstance(lists[0][0], str):
        return edit(sublist)
    else:
        print(lists[0][0])
        raise Exception("Elements of the editlist is neither list or tuple nor str")

def decode_edits(edits, mode='edit'):
    result = []
    if len(edits) == 0:
        return result
    for item in edits:
        if isinstance(item, edit):
            result.append(item)
        else:
            result = result + decode_edits(item, 'edit')

    assert(all([isinstance(item, edit) for item in result]))

    if mode == 'edit':
        return result
    elif mode == 'str':
        return [ op for e in result for op in e.serialize() ]
    else:
        raise AttributeError(f"Unsupported mode:{mode} in decode_edits!")

def llvmMutateWrap(srcEncIn, op:str, field1:str, field2:str, seed=None):
    """
    return returnCode, mutated and encoded source, edit with UID
    """
    mut_command = ['llvm-mutate']
    if op == 'c' or op == 'x':
        mut_command.extend(['-'+op, field1])
    else:
        mut_command.extend(['-'+op, field1 + ',' + field2])

    if seed is not None:
        mut_command.extend(['-d', '--seed', str(seed)])

    proc = subprocess.run(mut_command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          input=srcEncIn)

    if proc.returncode != 0 and proc.returncode != 1:
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
        except TypeError as err:
            print(proc.stderr.decode(), file=sys.stderr)
            with open('error.ll', 'w') as f:
                f.write(proc.stdout.decode())
            print(*mut_command)
            raise Exception("Could not understand the result from llvm-mutate") from err

    if proc.stderr.decode().find('no use') != -1:
        return 1, mutateSrc, edit(editUID)
    return 0, mutateSrc, edit(editUID)

def sort_serialized_edits(cmd):
    cmdlist = list(cmd)
    if all([isinstance(item, edit) for item in cmdlist]) is False:
        raise Exception("Only sorting on a per-edit basis!")

    c_cmd  = sorted([c for c in cmdlist if c[0][0] == '-c'])
    r_cmd  = sorted([c for c in cmdlist if c[0][0] == '-r'])
    i_cmd  = sorted([c for c in cmdlist if c[0][0] == '-i'])
    m_cmd  = sorted([c for c in cmdlist if c[0][0] == '-m'])
    s_cmd  = sorted([c for c in cmdlist if c[0][0] == '-s'])
    x_cmd  = sorted([c for c in cmdlist if c[0][0] == '-x'])
    op_cmd = sorted([c for c in cmdlist if c[0][0] == '-p'])

    cmdlist = s_cmd + m_cmd + i_cmd + r_cmd + c_cmd + x_cmd + op_cmd
    return cmdlist

def update_from_edits(idx, ind, resultList):
    '''Function for thread safe'''
    try:
        ind.update_src_from_edits()
        resultList[idx] = True
    except llvmIRrepRuntimeError:
        resultList[idx] = False

def edits_as_key(edits):
    serialized_edits = decode_edits(edits)
    serialized_edits = sort_serialized_edits(serialized_edits)

    return tuple(serialized_edits)

class llvmIRrepRuntimeError(RuntimeError):
    pass

class llvmIRrep():
    def __init__(self, srcEnc, mgpu, edits=None, mattr="+ptx70"):
        # default compilation argument to Nvidia pascal architecture.
        self.mgpu = mgpu
        self.mattr = mattr
        self._srcEnc = srcEnc
        self._lineSize = 0
        if edits is None:
            self._edits = []
            self._serialized_edits = []
            self._update_linesize()
        else:
            self._edits = list(edits)
            self._serialized_edits = decode_edits(edits)
            self.sort_serialized_edits()
            self.update_src_from_edits()

    @property
    def edits(self):
        return self._edits

    @property
    def serialized_edits(self):
        return self._serialized_edits

    @property
    def srcEnc(self):
        return self._srcEnc

    @property
    def lineSize(self):
        return self._lineSize

    @property
    def key(self):
        return tuple(self._serialized_edits)

    def __eq__(self, other):
        if isinstance(other, llvmIRrep):
            return self.key == other.key
        return False

    def __hash__(self):
        return hash(self.key)

    def _update_linesize(self):
        try:
            readline_proc = subprocess.run(['llvm-mutate', '-I'],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           input=self._srcEnc,
                                           check=True)
        except subprocess.CalledProcessError as err:
            print(err.stderr, file=sys.stderr)
            raise llvmIRrepRuntimeError('llvm-mutate error in calculating line size') from err

        self._lineSize = int(readline_proc.stderr.decode())

    def _update_src(self, srcEnc):
        self._srcEnc = srcEnc
        self._update_linesize()

    def update_edits(self, edits):
        self._edits = list(edits)
        self._serialized_edits = decode_edits(edits)
        self.sort_serialized_edits()

    def copy_from(self, other: 'llvmIRrep'):
        if self is other:
            return
        self.mgpu = other.mgpu
        self.mattr = other.mattr
        self._srcEnc = other.srcEnc
        self._lineSize = other.lineSize
        if len(other.serialized_edits) == 0 and len(other.edits) != 0:
            self.update_edits(other.edits)
        else:
            self._edits = list(other.edits)
            self._serialized_edits = list(other.serialized_edits)
    
    def update_src_from_edits(self, sweepEdits=False):
        assert(len(self.edits) != 0)
        if sweepEdits is False:
            proc = subprocess.run(['llvm-mutate', '--not_use_result'] + decode_edits(self.serialized_edits, 'str'),
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  input=self._srcEnc)
            if proc.returncode != 0:
                self._srcEnc = None
                raise llvmIRrepRuntimeError()
            mutateSrcEn = proc.stdout
        else:
            raise NotImplementedError
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
        self._update_src(mutateSrcEn)

    def sort_serialized_edits(self):
        if all([isinstance(item, edit) for item in self._serialized_edits]) is False:
            raise TypeError("Only sorting on a per-edit basis!")

        c_cmd  = sorted([c for c in self.serialized_edits if c[0][0] == '-c'])
        r_cmd  = sorted([c for c in self.serialized_edits if c[0][0] == '-r'])
        i_cmd  = sorted([c for c in self.serialized_edits if c[0][0] == '-i'])
        m_cmd  = sorted([c for c in self.serialized_edits if c[0][0] == '-m'])
        s_cmd  = sorted([c for c in self.serialized_edits if c[0][0] == '-s'])
        x_cmd  = sorted([c for c in self.serialized_edits if c[0][0] == '-x'])
        op_cmd = sorted([c for c in self.serialized_edits if c[0][0] == '-p'])

        self._serialized_edits = s_cmd + m_cmd + i_cmd + r_cmd + c_cmd + x_cmd + op_cmd

    def ptx(self, outf):
        proc = subprocess.run(['llc-'+__llvm_version__, "-march=nvptx64", "-mcpu="+self.mgpu, "-mattr="+self.mattr, '-o', outf],
                              stdout=subprocess.PIPE,
                              input=self._srcEnc)

        if proc.returncode != 0:
            print(proc.stderr.decode())
            print(proc.stdout.decode())
            raise llvmIRrepRuntimeError('llc error')
