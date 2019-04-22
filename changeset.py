import re
from typing import Dict, Optional, Union

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
    obj = changeset.newOp()

    def next(optObj):
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


def clear_op(op: Dict[str, Union[str, int]]) -> None:
    """Clean an Op object

    :param op: object to be cleared

    """
    op.opcode = ''
    op.chars = 0
    op.lines = 0
    op.attribs = ''


changeset.clearOp = clear_op


def new_op(optOpcode: Optional[str] = None) -> Dict[str, Union[str, int]]:
    """Create a new Op object

    :param optOpcode: the type operation of the Op object

    """
    return {'opcode': optOpcode.to_python() or '',
            'chars': 0,
            'lines': 0,
            'attribs': ''}


changeset.newOp = new_op