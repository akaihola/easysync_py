import re
import string
from typing import Dict, Optional, Union

from js2py.base import PyJs

from js_module import eval_js_module

changeset = eval_js_module('Changeset.js').exports


def parseNum(s: str) -> int:
    """Parse a number from string base 36

    :param s: string of the number in base 36
    :return: parsed number

    """
    if isinstance(s, PyJs):
        s = s.to_python()
    return int(s, 36)


changeset.parseNum = parseNum


BASE36_ALPHABET = string.digits + string.ascii_lowercase


def numToString(num: int) -> str:
    """Write a number in base 36 and return it as a string

    :param num: number to encode
    :return: number encoded as a base-36 string

    """
    num = num.to_python()
    base36 = ''
    while num:
        num, i = divmod(num, 36)
        base36 = BASE36_ALPHABET[i] + base36
    return base36 or BASE36_ALPHABET[0]


changeset.numToString = numToString


class OpIterator:
    def __init__(self, opsStr: str, optStartIndex: int):
        """Create an iterator which decodes string changeset operations

        :param opsStr: String encoding of the change operations to be performed
        :param optStartIndex: from where in the string should the iterator start
        :return: type object iterator

        """
        self.opsStr = opsStr.to_python()
        self.regex = re.compile(
            r'((?:\*[0-9a-z]+)*)(?:\|([0-9a-z]+))?([-+=])([0-9a-z]+)'
            r'|\?'
            r'|')
        self.startIndex = optStartIndex.to_python() or 0
        self.curIndex = self.startIndex
        self.prevIndex = self.curIndex
        self.regexResult = self.nextRegexMatch()
        self.obj = Op()

    @staticmethod
    def new_op(opsStr: str, optStartIndex: int):
        """JavaScript compatibility constructor"""
        return OpIterator(opsStr, optStartIndex)

    def nextRegexMatch(self):
        self.prevIndex = self.curIndex
        regex_lastIndex = self.curIndex
        result = self.regex.search(self.opsStr, pos=regex_lastIndex)
        self.curIndex = result.end()
        if result.group(0) == '?':
            changeset.error("Hit error opcode in op stream")
        return result

    def next(self, optObj: Optional['Op'] = None) -> 'Op':
        if isinstance(optObj, PyJs):
            optObj = optObj.to_python()
        op = optObj or self.obj
        if self.regexResult.group(0):
            op.attribs = self.regexResult.group(1)
            op.lines = changeset.parseNum(self.regexResult.group(2) or '0')
            op.opcode = self.regexResult.group(3)
            op.chars = changeset.parseNum(self.regexResult.group(4))
            self.regexResult = self.nextRegexMatch()
        else:
            op.clear()
        return op

    def hasNext(self):
        return bool(self.regexResult.group(0))

    def lastIndex(self):
        return self.prevIndex


changeset.opIterator = OpIterator.new_op


class Op:
    def __init__(self,
                 opcode: Optional[str] = None,
                 chars: int = 0,
                 lines: int = 0,
                 attribs: str = ''):
        """Create a new Op object

        :param opcode: the type operation of the Op object

        """
        self.opcode = opcode or ''
        self.chars = chars
        self.lines = lines
        self.attribs = attribs

    @staticmethod
    def new_op(optOpcode: Optional[str] = None) -> 'Op':
        """Create a new Op object from JavaScript

        :param optOpcode: the type operation of the Op object

        """
        return Op(optOpcode.to_python())

    def clear(self):
        """Clean an Op object"""
        self.opcode = ''
        self.chars = 0
        self.lines = 0
        self.attribs = ''

    def clone(self):
        """Clone an op"""
        return type(self)(self.opcode, self.chars, self.lines, self.attribs)

    def copy(self, op2):
        op2.opcode = self.opcode
        op2.chars = self.chars
        op2.lines = self.lines
        op2.attribs = self.attribs

    @staticmethod
    def copy_op(op1: 'Op', op2: 'Op'):
        """Copy the Op object to another one in JavaScript

        :param op1: src Op
        :param op2: dest Op

        """
        op2['opcode'] = op1['opcode']
        op2['chars'] = op1['chars']
        op2['lines'] = op1['lines']
        op2['attribs'] = op1['attribs']


changeset.newOp = Op.new_op
changeset.clearOp = Op.clear
changeset.cloneOp = Op.clone
changeset.copyOp = Op.copy_op


def unpack(cs: str) -> Dict[str, Union[int, str]]:
    if isinstance(cs, PyJs):
        cs = cs.to_python()
    headerRegex = re.compile(r'Z:([0-9a-z]+)([><])([0-9a-z]+)|')
    headerMatch = headerRegex.search(cs)
    if not headerMatch or not headerMatch.group(0):
        changeset.error(f'Not a changeset: {cs}')
    oldLen = parseNum(headerMatch.group(1))
    changeSign = 1 if headerMatch.group(2) == '>' else -1
    changeMag = parseNum(headerMatch.group(3))
    newLen = oldLen + changeSign * changeMag
    opsStart = len(headerMatch.group(0))
    opsEnd = cs.find("$")
    if opsEnd < 0:
        opsEnd = len(cs)
    return {'oldLen': oldLen,
            'newLen': newLen,
            'ops': cs[opsStart:opsEnd],
            'charBank': cs[opsEnd + 1:]}


changeset.unpack = unpack
