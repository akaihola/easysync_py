import re
from typing import Dict, Optional

from js_module import eval_js_module

changeset = eval_js_module('Changeset.js').exports


def op_iterator(opsStr: str, optStartIndex: int) -> Dict:
    """Create an iterator which decodes string changeset operations

    :param opsStr: String encoding of the change operations to be performed
    :param optStartIndex: from where in the string should the iterator start
    :return: type object iterator

    """
    opsStr = opsStr.to_python()
    regex = re.compile(
        r'((?:\*[0-9a-z]+)*)(?:\|([0-9a-z]+))?([-+=])([0-9a-z]+)'
        r'|\?'
        r'|')
    startIndex = optStartIndex.to_python() or 0
    curIndex = startIndex
    prevIndex = curIndex

    def nextRegexMatch():
        nonlocal curIndex, prevIndex
        prevIndex = curIndex
        regex_lastIndex = curIndex
        result = regex.search(opsStr, pos=regex_lastIndex)
        curIndex = result.end()
        if result.group(0) == '?':
            changeset.error("Hit error opcode in op stream")
        return result

    regexResult = nextRegexMatch()
    obj = Op()

    def next(optObj: Optional[Op]) -> Op:
        nonlocal regexResult
        op = optObj.to_python() or obj
        if regexResult.group(0):
            op.attribs = regexResult.group(1)
            op.lines = changeset.parseNum(regexResult.group(2) or 0)
            op.opcode = regexResult.group(3)
            op.chars = changeset.parseNum(regexResult.group(4))
            regexResult = nextRegexMatch()
        else:
            changeset.clearOp(op)
        return op

    def hasNext():
        return bool(regexResult.group(0))

    def lastIndex():
        return prevIndex

    return {'next': next,
            'hasNext': hasNext,
            'lastIndex': lastIndex}


changeset.opIterator = op_iterator


class Op:
    def __init__(self, opcode: Optional[str] = None):
        """Create a new Op object

        :param opcode: the type operation of the Op object

        """
        self.opcode = opcode or ''
        self.chars = 0
        self.lines = 0
        self.attribs = ''

    @staticmethod
    def new_op(optOpcode: Optional[str] = None) -> 'Op':
        """Create a new Op object from JavaScript

        :param optOpcode: the type operation of the Op object

        """
        return Op(optOpcode.to_python())


changeset.newOp = Op.new_op


def clear_op(op: Op) -> None:
    """Clean an Op object

    :param op: object to be cleared

    """
    op.opcode = ''
    op.chars = 0
    op.lines = 0
    op.attribs = ''


changeset.clearOp = clear_op


def clone_op(op: Op) -> None:
    """Clone an op

    :param op: Op to be cloned

    """
    return {'opcode': op.opcode,
            'chars': op.chars,
            'lines': op.lines,
            'attribs': op.attribs}


changeset.cloneOp = clone_op


def copy_op(op1: Op, op2: Op) -> None:
    """Copy op1 to op2

    :param op1: src Op
    :param op2: dest Op

    """
    op2['opcode'] = op1['opcode']
    op2['chars'] = op1['chars']
    op2['lines'] = op1['lines']
    op2['attribs'] = op1['attribs']


changeset.copyOp = copy_op
