import re
import string
from dataclasses import dataclass
from typing import Dict, Optional, Union

from js2py.base import HJs
from pkg_resources import resource_filename

from easysync_py.js_module import eval_js_module

JS_PATH = resource_filename('easysync_py', 'Changeset.js')
changeset = eval_js_module(JS_PATH).exports


class EasySyncError(Exception):
    pass


def error(msg: str):
    raise EasySyncError(msg)


changeset.error = HJs(error)


def assert_(b: bool, *msgParts):
    if not b:
        error(f'Failed assertion: {"".join(str(p) for p in msgParts)}')


setattr(changeset, 'assert', HJs(assert_))


def parseNum(s: str) -> int:
    """Parse a number from string base 36

    :param s: string of the number in base 36
    :return: parsed number

    """
    return int(s, 36)


changeset.parseNum = HJs(parseNum)


BASE36_ALPHABET = string.digits + string.ascii_lowercase


def numToString(num: int) -> str:
    """Write a number in base 36 and return it as a string

    :param num: number to encode
    :return: number encoded as a base-36 string

    """
    base36 = ''
    while num:
        num, i = divmod(num, 36)
        base36 = BASE36_ALPHABET[i] + base36
    return base36 or BASE36_ALPHABET[0]


changeset.numToString = HJs(numToString)


#changeset.toBaseTen = HJs(toBaseTen)
#changeset.oldLen = HJs(oldLen)
#changeset.newLen = HJs(newLen)


class OpIterator:
    def __init__(self, opsStr: str, optStartIndex: int = 0):
        """Create an iterator which decodes string changeset operations

        :param opsStr: String encoding of the change operations to be performed
        :param optStartIndex: from where in the string should the iterator
                              start
        :return: type object iterator

        """
        self.opsStr = opsStr
        self.regex = re.compile(
            r'((?:\*[0-9a-z]+)*)(?:\|([0-9a-z]+))?([-+=])([0-9a-z]+)'
            r'|\?'
            r'|')
        self.startIndex = optStartIndex or 0
        self.curIndex = self.startIndex
        self.prevIndex = self.curIndex
        self.regexResult = self.nextRegexMatch()
        self.obj = Op()

    def nextRegexMatch(self):
        self.prevIndex = self.curIndex
        regex_lastIndex = self.curIndex
        result = self.regex.search(self.opsStr, pos=regex_lastIndex)
        self.curIndex = result.end()
        if result.group(0) == '?':
            error('Hit error opcode in op stream')
        return result

    def next(self, optObj: Optional['Op'] = None) -> 'Op':
        op = optObj or self.obj
        if self.regexResult.group(0):
            op.attribs = self.regexResult.group(1)
            op.lines = parseNum(self.regexResult.group(2) or '0')
            op.opcode = self.regexResult.group(3)
            op.chars = parseNum(self.regexResult.group(4))
            self.regexResult = self.nextRegexMatch()
        else:
            op.clear()
        return op

    def __iter__(self):
        return self

    def __next__(self):
        op = Op()
        self.next(op)
        if not op.opcode:
            raise StopIteration()
        return op

    def hasNext(self):
        return bool(self.regexResult.group(0))

    def lastIndex(self):
        return self.prevIndex


changeset.opIterator = HJs(OpIterator)


@dataclass
class Op:
    """Create a new Op object

    :param opcode: the type operation of the Op object

    """
    opcode: Optional[str] = None
    chars: int = 0
    lines: int = 0
    attribs: str = ''

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
        op2.opcode = op1.opcode
        op2.chars = op1.chars
        op2.lines = op1.lines
        op2.attribs = op1.attribs


changeset.clearOp = HJs(Op.clear)
changeset.newOp = HJs(Op)
changeset.cloneOp = HJs(Op.clone)
changeset.copyOp = HJs(Op.copy_op)


# changeset.opString = HJs(opString)
# changeset.stringOp = HJs(stringOp)
# changeset.checkRep = HJs(checkRep)
# changeset.smartOpAssembler = HJs(smartOpAssembler)
# changeset.mergingOpAssembler = HJs(mergingOpAssembler)
# changeset.opAssembler = HJs(opAssembler)


class StringIterator:
    """A custom made String Iterator"""

    def __init__(self, text: str):
        """Create the string iterator

        :param text: String to be iterated over

        """
        self.text = text
        self.length = len(text)
        self.curIndex = 0
        # newLines is the number of `\n` between curIndex and len(str)
        self.newLines = text.count('\n')

    def newlines(self):
        return self.newLines

    def assertRemaining(self, n):
        assert_(n <= self.remaining(), "!(", n, " <= ", self.remaining(), ")")

    def take(self, n):
        self.assertRemaining(n)
        s = self.text[self.curIndex:self.curIndex + n]
        self.newLines -= s.count('\n')
        self.curIndex += n
        return s

    def peek(self, n):
        self.assertRemaining(n)
        return self.text[self.curIndex:self.curIndex + n]

    def peek_newline_count(self, n):
        """Count the number of newlines in the next ``n`` characters

        :param n: The number of characters to peek ahead
        :return: The number of newlines in the next ``n`` characters

        """
        return self.peek(n).count('\n')

    def skip(self, n):
        self.assertRemaining(n)
        self.curIndex += n

    def remaining(self):
        return self.length - self.curIndex


changeset.stringIterator = HJs(StringIterator)


class StringAssembler:
    """A custom made StringBuffer"""
    def __init__(self):
        self.pieces = []

    def append(self, x):
        self.pieces.append(str(x))

    def toString(self):
        return ''.join(self.pieces)


changeset.stringAssembler = HJs(StringAssembler)


# changeset.textLinesMutator = HJs(textLinesMutator)
# changeset.applyZip = HJs(applyZip)


HEADER_RE = re.compile(r'Z:([0-9a-z]+)([><])([0-9a-z]+)')


def unpack(cs: str) -> Dict[str, Union[int, str]]:
    """Unpack a string encoded Changeset into a proper Changeset object

    :param cs: String encoded Changeset
    :return: a Changeset data structure

    """
    match = HEADER_RE.search(cs)
    if not match:
        error(f'Not a changeset: {cs}')
    old_length = parseNum(match.group(1))
    change_sign = 1 if match.group(2) == '>' else -1
    change_mag = parseNum(match.group(3))
    newLen = old_length + change_sign * change_mag
    ops_start = len(match.group(0))
    ops_end = cs.find("$")
    if ops_end < 0:
        ops_end = len(cs)
    return {'oldLen': old_length,
            'newLen': newLen,
            'ops': cs[ops_start:ops_end],
            'charBank': cs[ops_end + 1:]}


changeset.unpack = HJs(unpack)


def pack(oldLen: int, newLen: int, opsStr: str, bank: str) -> str:
    """Pack a Changeset object into a string

    :param oldLen: Old length of the Changeset
    :param newLen: New length of the Changeset
    :param opsStr: String encoding of the changes to be made
    :param bank: Charbank of the Changeset
    :return: a Changeset string

    """
    lenDiff = newLen - oldLen
    lenDiffStr = (f'>{numToString(lenDiff)}' if lenDiff >= 0
                  else f'<{numToString(-lenDiff)}')
    return f'Z:{numToString(oldLen)}{lenDiffStr}{opsStr}${bank}'


changeset.pack = HJs(pack)


def applyToText(cs: str, text: str) -> str:
    """Applies a Changeset to a string

    :param cs: String encoded Changeset
    :param text: String to which a Changeset should be applied

    """
    unpacked = unpack(cs)
    assert_(len(text) == unpacked['oldLen'],
            'mismatched apply: ', len(text), ' / ', unpacked['oldLen'])
    csIter = OpIterator(unpacked['ops'])
    bankIter = StringIterator(unpacked['charBank'])
    strIter = StringIterator(text)
    assem = []
    for op in csIter:
        if op.opcode == '+':
            # op is + and op.lines 0:
            # -> no newlines must be in op.chars
            # op is + and op.lines >0:
            # -> op.chars must include op.lines newlines
            if op.lines != bankIter.peek_newline_count(op.chars):
                raise EasySyncError(f'newline count is wrong in op +; '
                                    f'cs:{cs} and text:{text}')
            assem.append(bankIter.take(op.chars))
        elif op.opcode == '-':
            # op is - and op.lines 0:
            # -> no newlines must be in the deleted string
            # op is - and op.lines >0:
            # -> op.lines newlines must be in the deleted string
            if op.lines != strIter.peek_newline_count(op.chars):
                raise EasySyncError(f'newline count is wrong in op -; '
                                    f'cs:{cs} and text:{text}')
            strIter.skip(op.chars)
        elif op.opcode == '=':
            # op is = and op.lines 0:
            # -> no newlines must be in the copied string
            # op is = and op.lines >0:
            # -> op.lines newlines must be in the copied string
            if op.lines != strIter.peek_newline_count(op.chars):
                raise EasySyncError('newline count is wrong in op =; '
                                    'cs:{cs} and text:{str}')
            assem.append(strIter.take(op.chars))
    assem.append(strIter.take(strIter.remaining()))
    return ''.join(assem)


changeset.applyToText = HJs(applyToText)


# changeset.mutateTextLines = HJs(mutateTextLines)
# changeset.composeAttributes = HJs(composeAttributes)
# changeset._slicerZipperFunc = HJs(_slicerZipperFunc)
# changeset.applyToAttribution = HJs(applyToAttribution)
# changeset.mutateAttributionLines = HJs(mutateAttributionLines)
# changeset.joinAttributionLines = HJs(joinAttributionLines)
# changeset.splitAttributionLines = HJs(splitAttributionLines)


SPLIT_TEXT_LINES_RE = re.compile(r'[^\n]*(?:\n|[^\n]$)')


def splitTextLines(text):
    lines = SPLIT_TEXT_LINES_RE.findall(text)
    return lines


changeset.splitTextLines = HJs(splitTextLines)


# changeset.compose = HJs(compose)
# changeset.attributeTester = HJs(attributeTester)
# changeset.identity = HJs(identity)
# changeset.makeSplice = HJs(makeSplice)
# changeset.toSplices = HJs(toSplices)
# changeset.characterRangeFollow = HJs(characterRangeFollow)
# changeset.moveOpsToNewPool = HJs(moveOpsToNewPool)
# changeset.makeAttribution = HJs(makeAttribution)
# changeset.eachAttribNumber = HJs(eachAttribNumber)
# changeset.filterAttribNumbers = HJs(filterAttribNumbers)
# changeset.mapAttribNumbers = HJs(mapAttribNumbers)
# changeset.makeAText = HJs(makeAText)
# changeset.applyToAText = HJs(applyToAText)
# changeset.cloneAText = HJs(cloneAText)
# changeset.copyAText = HJs(copyAText)
# changeset.appendATextToAssembler = HJs(appendATextToAssembler)
# changeset.prepareForWire = HJs(prepareForWire)
# changeset.isIdentity = HJs(isIdentity)
# changeset.opAttributeValue = HJs(opAttributeValue)
# changeset.attribsAttributeValue = HJs(attribsAttributeValue)
# changeset.builder = HJs(builder)
# changeset.makeAttribsString = HJs(makeAttribsString)
# changeset.subattribution = HJs(subattribution)
# changeset.inverse = HJs(inverse)
# changeset.follow = HJs(follow)
# changeset.followAttributes = HJs(followAttributes)
# changeset.composeWithDeletions = HJs(composeWithDeletions)
# changeset._slicerZipperFuncWithDeletions = HJs(_slicerZipperFuncWithDeletions)
