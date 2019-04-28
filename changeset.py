import re
import string
from typing import Dict, Optional, Union

from js2py.base import HJs

from js_module import eval_js_module

changeset = eval_js_module('Changeset.js').exports


class EasySyncError(Exception):
    pass


def error(msg: str):
    raise EasySyncError(msg)


changeset.error = HJs(error)


# changeset.assert = HJs(assert)


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
            changeset.error("Hit error opcode in op stream")
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

    def hasNext(self):
        return bool(self.regexResult.group(0))

    def lastIndex(self):
        return self.prevIndex


changeset.opIterator = HJs(OpIterator)


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
# changeset.stringIterator = HJs(stringIterator)
# changeset.stringAssembler = HJs(stringAssembler)
# changeset.textLinesMutator = HJs(textLinesMutator)
# changeset.applyZip = HJs(applyZip)


def unpack(cs: str) -> Dict[str, Union[int, str]]:
    """Unpack a string encoded Changeset into a proper Changeset object

    :param cs: String encoded Changeset
    :return: a Changeset data structure

    """
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


# changeset.applyToText = HJs(applyToText)
# changeset.mutateTextLines = HJs(mutateTextLines)
# changeset.composeAttributes = HJs(composeAttributes)
# changeset._slicerZipperFunc = HJs(_slicerZipperFunc)
# changeset.applyToAttribution = HJs(applyToAttribution)
# changeset.mutateAttributionLines = HJs(mutateAttributionLines)
# changeset.joinAttributionLines = HJs(joinAttributionLines)
# changeset.splitAttributionLines = HJs(splitAttributionLines)
# changeset.splitTextLines = HJs(splitTextLines)
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
