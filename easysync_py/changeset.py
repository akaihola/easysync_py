import re
import string
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

from js2py.base import HJs, JsObjectWrapper
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


OP_REGEX = re.compile(r'((?:\*[0-9a-z]+)*)(?:\|([0-9a-z]+))?([-+=])([0-9a-z]+)'
                      r'|\?'
                      r'|')


class OpIterator:
    def __init__(self, opsStr: str, optStartIndex: int = 0):
        """Create an iterator which decodes string changeset operations

        ..note:: :func:`iterate_ops` below is a Pythonic generator version of
                 this class. This class should be removed in the future.

        :param opsStr: String encoding of the change operations to be performed
        :param optStartIndex: from where in the string should the iterator
                              start
        :return: type object iterator

        """
        self.opsStr = opsStr
        self.startIndex = optStartIndex or 0
        self.curIndex = self.startIndex
        self.prevIndex = self.curIndex
        self.regexResult = self.nextRegexMatch()
        self.obj = Op()

    def nextRegexMatch(self):
        self.prevIndex = self.curIndex
        regex_lastIndex = self.curIndex
        result = OP_REGEX.search(self.opsStr, pos=regex_lastIndex)
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


def iterate_ops(ops: str, start_index: int = 0) -> Generator[Op, None, None]:
    """Pythonic version of ``OpIterator``

    A generator which decodes string changeset operations

    :param ops: String encoding of the change operations to be performed
    :param start_index: from where in the string should the generator start
    :return: a generator for :class:`Op` objects

    """
    cur_index = start_index or 0

    while True:
        match = OP_REGEX.search(ops, pos=cur_index)
        cur_index = match.end()
        if not match.group(0):
            break
        if match.group(0) == '?':
            error('Hit error opcode in op stream')
        opcode = match.group(3)
        if not opcode:
            break
        op = Op(opcode=opcode,
                attribs=match.group(1),
                lines=parseNum(match.group(2) or '0'),
                chars=parseNum(match.group(4)))
        yield op


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


class TextLinesMutator:
    def __init__(self, lines):
        """This class allows to iterate and modify texts which have several lines

        It is used for applying Changesets on arrays of lines
        Note from prev docs: "lines" need not be an array as long as it
        supports certain calls (lines_foo inside).

        Mutates lines, an array of strings, in place.
        Mutation operations have the same constraints as exports operations
        with respect to newlines, but not the other additional constraints
        (i.e. ins/del ordering, forbidden no-ops, non-mergeability, final
        newline). Can be used to mutate lists of strings where the last char
        of each string is not actually a newline, but for the purposes of ``N``
        and ``L`` values, the caller should pretend it is, and for things to
        work right in that case, the input to ``insert()`` should be a single
        line with no newlines.

        :param lines: The lines to mutate

        """
        if isinstance(lines, JsObjectWrapper):
            # Compatibility code for usage from JavaScript
            self._original = lines
            self.lines = list(lines)
        else:
            self._original = None
            self.lines = lines
        self.curSplice = [0, 0]
        self.inSplice = False
        # position in document after curSplice is applied:
        self.curLine = 0
        self.curCol = 0
        # invariant:
        #   if inSplice
        #   then curLine is in curSplice[0] + len(curSplice) - {2,3}
        #   and curLine >= curSplice[0]
        # invariant:
        #   if inSplice and (curLine >= curSplice[0] + len(curSplice) - 2))
        #   then curCol == 0

    def lines_applySplice(self, s):
        try:
            start, length, *insert = s
        except ValueError:
            breakpoint()
        self.lines[start:start + length] = insert
        if self._original is not None:
            # Compatibility code for usage from JavaScript
            self._original.splice(*s)

    def lines_toSource(self):
        return repr(self.lines)

    def lines_get(self, idx):
        return self.lines[idx]
        # can be unimplemented if removeLines's return value not needed

    def lines_slice(self, start, end):
        return self.lines[start:end]

    def lines_length(self):
        return len(self.lines)

    def enterSplice(self):
        self.curSplice[0] = self.curLine
        self.curSplice[1] = 0
        if self.curCol > 0:
            self.putCurLineInSplice()
        self.inSplice = True

    def leaveSplice(self):
        self.lines_applySplice(self.curSplice)
        self.curSplice[:] = [0, 0]
        self.inSplice = False

    def isCurLineInSplice(self):
        return self.curLine - self.curSplice[0] < len(self.curSplice) - 2

    def debugPrint(self, typ):
        print(f'{typ}: {self.curSplice!r} / '
              f'{self.curLine},{self.curCol} / '
              f'{self.lines!r}')

    def putCurLineInSplice(self):
        if not self.isCurLineInSplice():
            self.curSplice.append(self.lines[self.curSplice[0] +
                                             self.curSplice[1]])
            self.curSplice[1] += 1
        return 2 + self.curLine - self.curSplice[0]

    def skipLines(self, L, includeInSplice=False):
        if L:
            if includeInSplice:
                if not self.inSplice:
                    self.enterSplice()
                for i in range(L):
                    self.curCol = 0
                    self.putCurLineInSplice()
                    self.curLine += 1
            else:
                if self.inSplice:
                    if L > 1:
                        self.leaveSplice()
                    else:
                        self.putCurLineInSplice()
                self.curLine += L
                self.curCol = 0
            # print(f'{self.inSplice} / '
            #       f'{self.isCurLineInSplice()} / '
            #       f'{self.curSplice[0]} / '
            #       f'{self.curSplice[1]} / '
            #       f'{len(self.lines)}')
            # if (self.inSplice
            #         and not self.isCurLineInSplice()
            #         and self.curSplice[0] + self.curSplice[1] < len(self.lines):
            #     print("BLAH")
            #     self.putCurLineInSplice()
            # tests case foo in remove(), which isn't otherwise covered in
            # current impl
        # debugPrint('skip')

    def skip(self, N, L=None, includeInSplice=False):
        if N:
            if L:
                self.skipLines(L, includeInSplice)
            else:
                if includeInSplice and not self.inSplice:
                    self.enterSplice()
                if self.inSplice:
                    self.putCurLineInSplice()
                self.curCol += N
                # debugPrint("skip")

    def removeLines(self, L):
        removed = ''
        if L:
            if not self.inSplice:
                self.enterSplice()

            def nextKLinesText(k):
                m = self.curSplice[0] + self.curSplice[1]
                try:
                    return ''.join(self.lines[m:m + k])
                except TypeError:
                    breakpoint()

            if self.isCurLineInSplice():
                # print(self.curCol)
                if self.curCol == 0:
                    removed = self.curSplice[-1]
                    # print('FOO'); # case foo
                    self.curSplice.pop()
                    removed += nextKLinesText(L - 1)
                    self.curSplice[1] += L - 1
                else:
                    removed = nextKLinesText(L - 1)
                    self.curSplice[1] += L - 1
                    sline = len(self.curSplice) - 1
                    removed = self.curSplice[sline][self.curCol:] + removed
                    self.curSplice[sline] = (
                            self.curSplice[sline][:self.curCol] +
                            self.lines[self.curSplice[0] + self.curSplice[1]])
                    self.curSplice[1] += 1
            else:
                removed = nextKLinesText(L)
                self.curSplice[1] += L
            # debugPrint("remove")
        return removed

    def remove(self, N, L, verify=None):
        removed = ''
        if N:
            if L:
                return self.removeLines(L)
            else:
                if not self.inSplice:
                    self.enterSplice()
                sline = self.putCurLineInSplice()
                removed = self.curSplice[sline][self.curCol:self.curCol + N]
                self.curSplice[sline] = (
                        self.curSplice[sline][:self.curCol] +
                        self.curSplice[sline][self.curCol + N:])
                # debugPrint("remove")
        if verify is not None and removed != verify:
            raise EasySyncError('Expected to remove {!r}, but removed {!r} '
                                'instead'.format(verify, removed))
        return removed

    def insert(self, text, L=None):
        if text:
            if not self.inSplice:
                self.enterSplice()
            if L:
                newLines = splitTextLines(text)
                if self.isCurLineInSplice():
                    # if self.curCol == 0:
                    #     self.curSplice.pop()
                    #     self.curSplice[1] -= 1
                    #     self.curSplice.extend(newLines)
                    #     self.curLine += len(newLines)
                    # else:
                    theLine = self.curSplice[-1]
                    lineCol = self.curCol
                    self.curSplice[-1] = theLine[:lineCol] + newLines[0]
                    self.curLine += 1
                    newLines.pop(0)
                    self.curSplice.extend(newLines)
                    self.curLine += len(newLines)
                    self.curSplice.append(theLine[lineCol:])
                    self.curCol = 0
                else:
                    self.curSplice.extend(newLines)
                    self.curLine += len(newLines)
            else:
                sline = self.putCurLineInSplice()
                self.curSplice[sline] = (self.curSplice[sline][:self.curCol] +
                                         text +
                                         self.curSplice[sline][self.curCol:])
                self.curCol += len(text)
            # debugPrint('insert')

    def hasMore(self):
        # print(f'{len(self.lines)} / '
        #       f'{self.inSplice} / '
        #       f'{len(self.curSplice) - 2} / '
        #       f'{self.curSplice[1])}')
        docLines = len(self.lines)
        if self.inSplice:
            docLines += len(self.curSplice) - 2 - self.curSplice[1]
        return self.curLine < docLines

    def close(self):
        if self.inSplice:
            self.leaveSplice()
        # debugPrint("close")


changeset.textLinesMutator = HJs(TextLinesMutator)


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
    :return: The resulting text after applying the Changeset

    """
    unpacked = unpack(cs)
    assert_(len(text) == unpacked['oldLen'],
            'mismatched apply: ', len(text), ' / ', unpacked['oldLen'])
    ops = iterate_ops(unpacked['ops'])
    bankIter = StringIterator(unpacked['charBank'])
    strIter = StringIterator(text)
    assem = []
    for op in ops:
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


def mutateTextLines(cs: str, lines: List[str]) -> None:
    """Apply a changeset on an array of lines

    :param cs: the changeset to be applied
    :param lines: The lines to which the changeset needs to be applied

    """
    unpacked = unpack(cs)
    ops = iterate_ops(unpacked['ops'])
    bankIter = StringIterator(unpacked['charBank'])
    mut = TextLinesMutator(lines)
    for op in ops:
        if op.opcode == '+':
            mut.insert(bankIter.take(op.chars), op.lines)
        elif op.opcode == '-':
            mut.remove(op.chars, op.lines)
        elif op.opcode == '=':
            mut.skip(op.chars, op.lines, bool(op.attribs))
    mut.close()


changeset.mutateTextLines = HJs(mutateTextLines)


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


def parse_insert_args(insert_chars, insert_num_lines=None):
    return insert_chars, insert_num_lines


def parse_skip_args(include_in_splice=False, check=None):
    return check


def parse_remove_args(check=None):
    return check


def parse_remove_skip_args(opcode, num_chars, num_lines=0, *args):
    if opcode == 'remove':
        return num_chars, num_lines, parse_remove_args(*args)
    else:
        return num_chars, num_lines, parse_skip_args(*args)


def ops_to_mutations(ops: List[Op], char_bank: str) -> List[List]:
    char_index = 0
    for op in ops:
        if op.opcode == '+':
            text = char_bank[char_index:char_index + op.chars]
            char_index += op.chars
            mutation = ['insert', text]
            num_lines = text.count('\n')
            if num_lines:
                mutation.append(num_lines)
        elif op.opcode == '-':
            mutation = ['remove', op.chars, op.lines]
            if hasattr(op, 'check'):
                mutation.append(op.check)
        elif op.opcode == '=':
            mutation = ['skip', op.chars]
            if op.lines:
                mutation.append(op.lines)
        else:
            raise EasySyncError(f'Invalid opcode "{op.opcode}"')
        yield mutation


LineRange = Tuple[int, int]
OpName = str
OpDetails = Tuple[int, int, str]
Change = Tuple[LineRange, LineRange, OpName, OpDetails]
Changes = List[Change]


def mutate(original: List[str],
           ops: Iterable[Op],
           char_bank: str) -> Tuple[List[str], Changes]:
    mutated = []
    changes = []
    orig_linenum = 0
    orig_column = 0
    char_index = 0

    def copy(chars_to_copy: str):
        if not mutated or mutated[-1].endswith('\n'):
            mutated.append(chars_to_copy)
        else:
            mutated[-1] += chars_to_copy

    for op in ops:
        prev_orig_linenum = orig_linenum
        prev_len_mutated = len(mutated)
        was_inside_line = (prev_len_mutated
                           and not mutated[-1].endswith('\n'))
        if op.opcode == '+':
            chars_processed = char_bank[char_index:char_index + op.chars]
            char_index += op.chars
            num_lines_inserted = chars_processed.count('\n')
            if op.lines is not None:
                assert num_lines_inserted == op.lines
            for line in splitTextLines(chars_processed):
                copy(line)
            if chars_processed.endswith('\n') and not was_inside_line:
                end_affected_original_linenum = orig_linenum
            else:
                end_affected_original_linenum = orig_linenum + 1
            if was_inside_line:
                begin_affected_new_linenum = prev_len_mutated - 1
            else:
                begin_affected_new_linenum = prev_len_mutated
            affected_new_line_range = begin_affected_new_linenum, len(mutated)
            print(f'insert {chars_processed!r}, '
                  f'num_lines={num_lines_inserted}')
        elif op.opcode in ['=', '-']:
            chars_left_to_process = op.chars
            lines_left_to_process = op.lines
            check = getattr(op, 'check', None)
            chars_processed = []
            prev_orig_column = orig_column
            while chars_left_to_process:
                orig_line = original[orig_linenum]
                num_chars_on_line = min(len(orig_line) - orig_column,
                                        chars_left_to_process)
                prev_orig_column = orig_column
                orig_column += num_chars_on_line
                chars_to_process = orig_line[prev_orig_column:orig_column]
                chars_processed.append(chars_to_process)
                lines_left_to_process -= chars_to_process.count('\n')
                print(f'{op.opcode} {chars_to_process!r}, '
                      f'num_lines == {lines_left_to_process}')
                assert lines_left_to_process >= 0
                if op.opcode == '=':
                    copy(chars_to_process)
                chars_left_to_process -= num_chars_on_line
                if orig_column == len(orig_line):
                    orig_column = 0
                    orig_linenum += 1
            assert lines_left_to_process == 0
            end_affected_original_linenum = orig_linenum + 1
            if op.opcode == '=':
                affected_new_line_range = None
            else:
                affected_new_line_range = (
                    prev_len_mutated - (1 if was_inside_line else 0),
                    prev_len_mutated + (0 if was_inside_line else 1))
            if check is not None:
                if ''.join(chars_processed) != check:
                    raise EasySyncError(f'Expected to {op.opcode} {check!r}, '
                                        f'but processed '
                                        f'{chars_processed!r}')
        else:
            raise EasySyncError(f'Invalid opcode "{op.opcode}"')
        affected_original_line_range = (prev_orig_linenum,
                                        end_affected_original_linenum)
        if affected_new_line_range:
            if op.opcode in ['-', '=']:
                chars_processed_detail = ''.join(chars_processed)
            else:
                chars_processed_detail = chars_processed
            op_details = op.chars, op.lines, chars_processed_detail
            changes.append((affected_original_line_range,
                            affected_new_line_range,
                            op.opcode, op_details))
    while orig_linenum < len(original):
        orig_line = original[orig_linenum]
        if orig_column < len(orig_line):
            copy(orig_line[orig_column:])
        orig_column = 0
        orig_linenum += 1

    return mutated, changes
