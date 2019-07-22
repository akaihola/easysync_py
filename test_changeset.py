from textwrap import dedent
from typing import List, Tuple

import pytest
from js2py.base import to_python, PyJsString

import easysync_py.changeset as py

js = py.changeset


@pytest.mark.parametrize(
    'old_len, new_len, ops_str, bank, expect',
    [(0, 0, 'opsStr', 'bank', "Z:0>0opsStr$bank"),
     (0, 5, 'opsStr', 'bank', "Z:0>5opsStr$bank"),
     (5, 0, 'opsStr', 'bank', "Z:5<5opsStr$bank"),
     (2, 4, 'opsStr', 'bank', "Z:2>2opsStr$bank"),
     (4, 2, 'opsStr', 'bank', "Z:4<2opsStr$bank"),
     (42, 42, 'opsStr', 'bank', "Z:16>0opsStr$bank")]
)
def test_pack(old_len, new_len, ops_str, bank, expect):
    py_packed = py.pack(old_len, new_len, ops_str, bank)
    js_packed = to_python(js.pack(old_len, new_len,
                                  PyJsString(ops_str), PyJsString(bank)))
    assert js_packed == expect
    assert py_packed == expect


TEXT = dedent('''\
    Adding first line.
    Adding second line.
    Duplicating first and second lines:
    Adding first line.
    Adding second line.
    Deleting everything but the last line.
''')

CHANGESETS = [
    ('Z:1>1*0+1$A',
     {'oldLen': 1, 'newLen': 2, 'ops': '*0+1', 'charBank': 'A'},
     TEXT[:0], TEXT[:1]),
    ('Z:2>3=1*0+3$ddi',
     {'oldLen': 2, 'newLen': 5, 'ops': '=1*0+3', 'charBank': 'ddi'},
     TEXT[:1], TEXT[:4]),
    ('Z:5>3=4*0+3$ng ',
     {'oldLen': 5, 'newLen': 8, 'ops': '=4*0+3', 'charBank': 'ng '},
     TEXT[:4], TEXT[:7]),
    ('Z:8>8=7*0+8$first li',
     {'oldLen': 8, 'newLen': 16, 'ops': '=7*0+8', 'charBank': 'first li'},
     TEXT[:7], TEXT[:15]),
    ('Z:g>4=f*0|1+4$ne.\n',
     {'oldLen': 16, 'newLen': 20, 'ops': '=f*0|1+4', 'charBank': 'ne.\n'},
     TEXT[:15], TEXT[:19]),
    ('Z:k>1|1=j*0+1$A',
     {'oldLen': 20, 'newLen': 21, 'ops': '|1=j*0+1', 'charBank': 'A'},
     TEXT[:19], TEXT[:20]),
    ('Z:l>1|1=j=1*0+1$d',
     {'oldLen': 21, 'newLen': 22, 'ops': '|1=j=1*0+1', 'charBank': 'd'},
     TEXT[:20], TEXT[:21]),
    ('Z:m>5|1=j=2*0+5$ding ',
     {'oldLen': 22, 'newLen': 27, 'ops': '|1=j=2*0+5', 'charBank': 'ding '},
     TEXT[:21], TEXT[:26]),
    ('Z:r>6|1=j=7*0+6$second',
     {'oldLen': 27, 'newLen': 33, 'ops': '|1=j=7*0+6', 'charBank': 'second'},
     TEXT[:26], TEXT[:32]),
    ('Z:x>6|1=j=d*0+6$ line.',
     {'oldLen': 33, 'newLen': 39, 'ops': '|1=j=d*0+6', 'charBank': ' line.'},
     TEXT[:32], TEXT[:38]),
    ('Z:13>1|1=j=j*0|1+1$\n',
     {'oldLen': 39, 'newLen': 40, 'ops': '|1=j=j*0|1+1', 'charBank': '\n'},
     TEXT[:38], TEXT[:39]),
    ('Z:14>1|2=13*0+1$D',
     {'oldLen': 40, 'newLen': 41, 'ops': '|2=13*0+1', 'charBank': 'D'},
     TEXT[:39], TEXT[:40]),
    ('Z:15>4|2=13=1*0+4$upli',
     {'oldLen': 41, 'newLen': 45, 'ops': '|2=13=1*0+4', 'charBank': 'upli'},
     TEXT[:40], TEXT[:44]),
    ('Z:19>6|2=13=5*0+6$cating',
     {'oldLen': 45, 'newLen': 51, 'ops': '|2=13=5*0+6', 'charBank': 'cating'},
     TEXT[:44], TEXT[:50]),
    ('Z:1f>5|2=13=b*0+5$ firs',
     {'oldLen': 51, 'newLen': 56, 'ops': '|2=13=b*0+5', 'charBank': ' firs'},
     TEXT[:50], TEXT[:55]),
    ('Z:1k>7|2=13=g*0+7$t and s',
     {'oldLen': 56, 'newLen': 63, 'ops': '|2=13=g*0+7', 'charBank': 't and s'},
     TEXT[:55], TEXT[:62]),
    ('Z:1r>1|2=13=n*0+1$e',
     {'oldLen': 63, 'newLen': 64, 'ops': '|2=13=n*0+1', 'charBank': 'e'},
     TEXT[:62], TEXT[:63]),
    ('Z:1s>7|2=13=o*0+7$cond li',
     {'oldLen': 64, 'newLen': 71, 'ops': '|2=13=o*0+7', 'charBank': 'cond li'},
     TEXT[:63], TEXT[:70]),
    ('Z:1z>9|2=13=v*0|1+5*0+4$nes:\n    ',
     {'oldLen': 71, 'newLen': 80, 'ops': '|2=13=v*0|1+5*0+4', 'charBank': 'nes:\n    '},
     TEXT[:70], TEXT[:79]),
    ('Z:28<4|3=23-4$',
     {'oldLen': 80, 'newLen': 76, 'ops': '|3=23-4', 'charBank': ''},
     'Adding first line.\nAdding second line.\nDuplicating first and second lines:\n    ', 'Adding first line.\nAdding second line.\nDuplicating first and second lines:\n'),
    ('Z:24>13|3=23*0|2+13$Adding first line.\nAdding second line.\n',
     {'oldLen': 76, 'newLen': 115, 'ops': '|3=23*0|2+13', 'charBank': 'Adding first line.\nAdding second line.\n'},
     TEXT[:75], TEXT[:114]),
    ('Z:37>1|5=36*0+1$D',
     {'oldLen': 115, 'newLen': 116, 'ops': '|5=36*0+1', 'charBank': 'D'},
     TEXT[:114], TEXT[:115]),
    ('Z:38>4|5=36=1*0+4$elet',
     {'oldLen': 116, 'newLen': 120, 'ops': '|5=36=1*0+4', 'charBank': 'elet'},
     TEXT[:115], TEXT[:119]),
    ('Z:3c>3|5=36=5*0+3$ing',
     {'oldLen': 120, 'newLen': 123, 'ops': '|5=36=5*0+3', 'charBank': 'ing'},
     TEXT[:119], TEXT[:122]),
    ('Z:3f>1|5=36=8*0+1$ ',
     {'oldLen': 123, 'newLen': 124, 'ops': '|5=36=8*0+1', 'charBank': ' '},
     TEXT[:122], TEXT[:123]),
    ('Z:3g>5|5=36=9*0+5$every',
     {'oldLen': 124, 'newLen': 129, 'ops': '|5=36=9*0+5', 'charBank': 'every'},
     TEXT[:123], TEXT[:128]),
    ('Z:3l>5|5=36=e*0+5$thing',
     {'oldLen': 129, 'newLen': 134, 'ops': '|5=36=e*0+5', 'charBank': 'thing'},
     TEXT[:128], TEXT[:133]),
    ('Z:3q>5|5=36=j*0+5$ but ',
     {'oldLen': 134, 'newLen': 139, 'ops': '|5=36=j*0+5', 'charBank': ' but '},
     TEXT[:133], TEXT[:138]),
    ('Z:3v>5|5=36=o*0+5$the l',
     {'oldLen': 139, 'newLen': 144, 'ops': '|5=36=o*0+5', 'charBank': 'the l'},
     TEXT[:138], TEXT[:143]),
    ('Z:40>6|5=36=t*0+6$ast li',
     {'oldLen': 144, 'newLen': 150, 'ops': '|5=36=t*0+6', 'charBank': 'ast li'},
     TEXT[:143], TEXT[:149]),
    ('Z:46>2|5=36=z*0+2$ne',
     {'oldLen': 150, 'newLen': 152, 'ops': '|5=36=z*0+2', 'charBank': 'ne'},
     TEXT[:149], TEXT[:151]),
    ('Z:48>1|5=36=11*0+1$.',
     {'oldLen': 152, 'newLen': 153, 'ops': '|5=36=11*0+1', 'charBank': '.'},
     TEXT[:151], TEXT[:152]),
    ('Z:49<34|4-2m-j*0|1+1$\n',
     {'oldLen': 153, 'newLen': 41, 'ops': '|4-2m-j*0|1+1', 'charBank': '\n'},
     TEXT[:152], TEXT[:40]),
    ('Z:15<1|1=1|1-1$',
     {'oldLen': 41, 'newLen': 40, 'ops': '|1=1|1-1', 'charBank': ''},
     '\n\nDeleting everything but the last line.', '\nDeleting everything but the last line.'),
    ('Z:14<1|1-1$',
     {'oldLen': 40, 'newLen': 39, 'ops': '|1-1', 'charBank': ''},
     '\nDeleting everything but the last line.', 'Deleting everything but the last line.')]


@pytest.mark.skip
def test_generate_changesets():
    text = ''
    for changeset, _, _, _ in CHANGESETS:
        unpacked = py.unpack(changeset)
        old_len = len(text)
        old_text = text
        new_text = py.changeset.applyToText(changeset, text + '\n')
        assert new_text[-1] == '\n'
        new_text = new_text[:-1]
        new_len = len(new_text)
        if TEXT.startswith(text):
            print(f'({changeset!r},\n'
                  f' {unpacked!r},\n'
                  f' TEXT[:{old_len}], TEXT[:{new_len}]),')
        else:
            print(f'({changeset!r},\n'
                  f' {unpacked!r},\n'
                  f' {old_text!r}, {new_text!r}),')
        text = new_text


@pytest.mark.skip
def test_generate_ops():
    """Helper used for converting original test data"""
    for _, unpacked, _, _ in CHANGESETS:
        ops = list(py.iterate_ops(unpacked['ops']))
        print(f'({unpacked["ops"]!r}, {ops!r}),')


@pytest.mark.parametrize('changeset, expect',
                         [(cs, expect) for cs, expect, _, _ in CHANGESETS])
def test_unpack(changeset, expect):
    unpacked = py.unpack(changeset)
    assert unpacked == expect


OPS = [('*0+1', [py.Op(opcode='+', chars=1, lines=0, attribs='*0')]),
       ('=1*0+3', [py.Op(opcode='=', chars=1, lines=0, attribs=''),
                   py.Op(opcode='+', chars=3, lines=0, attribs='*0')]),
       ('=4*0+3', [py.Op(opcode='=', chars=4, lines=0, attribs=''),
                   py.Op(opcode='+', chars=3, lines=0, attribs='*0')]),
       ('=7*0+8', [py.Op(opcode='=', chars=7, lines=0, attribs=''),
                   py.Op(opcode='+', chars=8, lines=0, attribs='*0')]),
       ('=f*0|1+4', [py.Op(opcode='=', chars=15, lines=0, attribs=''),
                     py.Op(opcode='+', chars=4, lines=1, attribs='*0')]),
       ('|1=j*0+1', [py.Op(opcode='=', chars=19, lines=1, attribs=''),
                     py.Op(opcode='+', chars=1, lines=0, attribs='*0')]),
       ('|1=j=1*0+1', [py.Op(opcode='=', chars=19, lines=1, attribs=''),
                       py.Op(opcode='=', chars=1, lines=0, attribs=''),
                       py.Op(opcode='+', chars=1, lines=0, attribs='*0')]),
       ('|1=j=2*0+5', [py.Op(opcode='=', chars=19, lines=1, attribs=''),
                       py.Op(opcode='=', chars=2, lines=0, attribs=''),
                       py.Op(opcode='+', chars=5, lines=0, attribs='*0')]),
       ('|1=j=7*0+6', [py.Op(opcode='=', chars=19, lines=1, attribs=''),
                       py.Op(opcode='=', chars=7, lines=0, attribs=''),
                       py.Op(opcode='+', chars=6, lines=0, attribs='*0')]),
       ('|1=j=d*0+6', [py.Op(opcode='=', chars=19, lines=1, attribs=''),
                       py.Op(opcode='=', chars=13, lines=0, attribs=''),
                       py.Op(opcode='+', chars=6, lines=0, attribs='*0')]),
       ('|1=j=j*0|1+1', [py.Op(opcode='=', chars=19, lines=1, attribs=''),
                         py.Op(opcode='=', chars=19, lines=0, attribs=''),
                         py.Op(opcode='+', chars=1, lines=1, attribs='*0')]),
       ('|2=13*0+1', [py.Op(opcode='=', chars=39, lines=2, attribs=''),
                      py.Op(opcode='+', chars=1, lines=0, attribs='*0')]),
       ('|2=13=1*0+4', [py.Op(opcode='=', chars=39, lines=2, attribs=''),
                        py.Op(opcode='=', chars=1, lines=0, attribs=''),
                        py.Op(opcode='+', chars=4, lines=0, attribs='*0')]),
       ('|2=13=5*0+6', [py.Op(opcode='=', chars=39, lines=2, attribs=''),
                        py.Op(opcode='=', chars=5, lines=0, attribs=''),
                        py.Op(opcode='+', chars=6, lines=0, attribs='*0')]),
       ('|2=13=b*0+5', [py.Op(opcode='=', chars=39, lines=2, attribs=''),
                        py.Op(opcode='=', chars=11, lines=0, attribs=''),
                        py.Op(opcode='+', chars=5, lines=0, attribs='*0')]),
       ('|2=13=g*0+7', [py.Op(opcode='=', chars=39, lines=2, attribs=''),
                        py.Op(opcode='=', chars=16, lines=0, attribs=''),
                        py.Op(opcode='+', chars=7, lines=0, attribs='*0')]),
       ('|2=13=n*0+1', [py.Op(opcode='=', chars=39, lines=2, attribs=''),
                        py.Op(opcode='=', chars=23, lines=0, attribs=''),
                        py.Op(opcode='+', chars=1, lines=0, attribs='*0')]),
       ('|2=13=o*0+7', [py.Op(opcode='=', chars=39, lines=2, attribs=''),
                        py.Op(opcode='=', chars=24, lines=0, attribs=''),
                        py.Op(opcode='+', chars=7, lines=0, attribs='*0')]),
       ('|2=13=v*0|1+5*0+4', [py.Op(opcode='=', chars=39, lines=2, attribs=''),
                              py.Op(opcode='=', chars=31, lines=0, attribs=''),
                              py.Op(opcode='+', chars=5, lines=1, attribs='*0'),
                              py.Op(opcode='+', chars=4, lines=0, attribs='*0')]),
       ('|3=23-4', [py.Op(opcode='=', chars=75, lines=3, attribs=''),
                    py.Op(opcode='-', chars=4, lines=0, attribs='')]),
       ('|3=23*0|2+13', [py.Op(opcode='=', chars=75, lines=3, attribs=''),
                         py.Op(opcode='+', chars=39, lines=2, attribs='*0')]),
       ('|5=36*0+1', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                      py.Op(opcode='+', chars=1, lines=0, attribs='*0')]),
       ('|5=36=1*0+4', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                        py.Op(opcode='=', chars=1, lines=0, attribs=''),
                        py.Op(opcode='+', chars=4, lines=0, attribs='*0')]),
       ('|5=36=5*0+3', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                        py.Op(opcode='=', chars=5, lines=0, attribs=''),
                        py.Op(opcode='+', chars=3, lines=0, attribs='*0')]),
       ('|5=36=8*0+1', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                        py.Op(opcode='=', chars=8, lines=0, attribs=''),
                        py.Op(opcode='+', chars=1, lines=0, attribs='*0')]),
       ('|5=36=9*0+5', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                        py.Op(opcode='=', chars=9, lines=0, attribs=''),
                        py.Op(opcode='+', chars=5, lines=0, attribs='*0')]),
       ('|5=36=e*0+5', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                        py.Op(opcode='=', chars=14, lines=0, attribs=''),
                        py.Op(opcode='+', chars=5, lines=0, attribs='*0')]),
       ('|5=36=j*0+5', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                        py.Op(opcode='=', chars=19, lines=0, attribs=''),
                        py.Op(opcode='+', chars=5, lines=0, attribs='*0')]),
       ('|5=36=o*0+5', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                        py.Op(opcode='=', chars=24, lines=0, attribs=''),
                        py.Op(opcode='+', chars=5, lines=0, attribs='*0')]),
       ('|5=36=t*0+6', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                        py.Op(opcode='=', chars=29, lines=0, attribs=''),
                        py.Op(opcode='+', chars=6, lines=0, attribs='*0')]),
       ('|5=36=z*0+2', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                        py.Op(opcode='=', chars=35, lines=0, attribs=''),
                        py.Op(opcode='+', chars=2, lines=0, attribs='*0')]),
       ('|5=36=11*0+1', [py.Op(opcode='=', chars=114, lines=5, attribs=''),
                         py.Op(opcode='=', chars=37, lines=0, attribs=''),
                         py.Op(opcode='+', chars=1, lines=0, attribs='*0')]),
       ('|4-2m-j*0|1+1', [py.Op(opcode='-', chars=94, lines=4, attribs=''),
                          py.Op(opcode='-', chars=19, lines=0, attribs=''),
                          py.Op(opcode='+', chars=1, lines=1, attribs='*0')]),
       ('|1=1|1-1', [py.Op(opcode='=', chars=1, lines=1, attribs=''),
                     py.Op(opcode='-', chars=1, lines=1, attribs='')]),
       ('|1-1', [py.Op(opcode='-', chars=1, lines=1, attribs='')])]


@pytest.mark.parametrize('ops, expect', OPS)
def test_op_iterator(ops, expect):
    op_iterator = py.OpIterator(ops)
    result = []
    for op in op_iterator:
        result.append(op)

    assert result == expect


@pytest.mark.parametrize('ops, expect', OPS)
def test_iterate_ops(ops, expect):
    result = list(py.iterate_ops(ops))

    assert result == expect


@pytest.mark.parametrize(
    'changeset, old_text, expect',
    [('Z:1>1*0+1$A', TEXT[:0], TEXT[:1]),
     ('Z:2>3=1*0+3$ddi', TEXT[:1], TEXT[:4]),
     ('Z:5>3=4*0+3$ng ', TEXT[:4], TEXT[:7]),
     ('Z:8>8=7*0+8$first li', TEXT[:7], TEXT[:15]),
     ('Z:g>4=f*0|1+4$ne.\n', TEXT[:15], TEXT[:19]),
     ('Z:k>1|1=j*0+1$A', TEXT[:19], TEXT[:20]),
     ('Z:l>1|1=j=1*0+1$d', TEXT[:20], TEXT[:21]),
     ('Z:m>5|1=j=2*0+5$ding ', TEXT[:21], TEXT[:26]),
     ('Z:r>6|1=j=7*0+6$second', TEXT[:26], TEXT[:32]),
     ('Z:x>6|1=j=d*0+6$ line.', TEXT[:32], TEXT[:38]),
     ('Z:13>1|1=j=j*0|1+1$\n', TEXT[:38], TEXT[:39]),
     ('Z:14>1|2=13*0+1$D', TEXT[:39], TEXT[:40]),
     ('Z:15>4|2=13=1*0+4$upli', TEXT[:40], TEXT[:44]),
     ('Z:19>6|2=13=5*0+6$cating', TEXT[:44], TEXT[:50]),
     ('Z:1f>5|2=13=b*0+5$ firs', TEXT[:50], TEXT[:55]),
     ('Z:1k>7|2=13=g*0+7$t and s', TEXT[:55], TEXT[:62]),
     ('Z:1r>1|2=13=n*0+1$e', TEXT[:62], TEXT[:63]),
     ('Z:1s>7|2=13=o*0+7$cond li', TEXT[:63], TEXT[:70]),
     #('Z:1z>9|2=13=v*0|1+5*0+4$nes:\n    ', TEXT[:70], TEXT[:79]),
     ('Z:28<4|3=23-4$', 'Adding first line.\nAdding second line.\nDuplicating first and second lines:\n    ', 'Adding first line.\nAdding second line.\nDuplicating first and second lines:\n'),
     ('Z:24>13|3=23*0|2+13$Adding first line.\nAdding second line.\n', TEXT[:75], TEXT[:114]),
     ('Z:37>1|5=36*0+1$D', TEXT[:114], TEXT[:115]),
     ('Z:38>4|5=36=1*0+4$elet', TEXT[:115], TEXT[:119]),
     ('Z:3c>3|5=36=5*0+3$ing', TEXT[:119], TEXT[:122]),
     ('Z:3f>1|5=36=8*0+1$ ', TEXT[:122], TEXT[:123]),
     ('Z:3g>5|5=36=9*0+5$every', TEXT[:123], TEXT[:128]),
     ('Z:3l>5|5=36=e*0+5$thing', TEXT[:128], TEXT[:133]),
     ('Z:3q>5|5=36=j*0+5$ but ', TEXT[:133], TEXT[:138]),
     ('Z:3v>5|5=36=o*0+5$the l', TEXT[:138], TEXT[:143]),
     ('Z:40>6|5=36=t*0+6$ast li', TEXT[:143], TEXT[:149]),
     ('Z:46>2|5=36=z*0+2$ne', TEXT[:149], TEXT[:151]),
     ('Z:48>1|5=36=11*0+1$.', TEXT[:151], TEXT[:152]),
     #('Z:49<34|4-2m-j*0|1+1$\n', TEXT[:152], TEXT[:40]),
     ('Z:15<1|1=1|1-1$', '\n\nDeleting everything but the last line.', '\nDeleting everything but the last line.'),
     ('Z:14<1|1-1$', '\nDeleting everything but the last line.', 'Deleting everything but the last line.'),])
def test_editing(changeset, old_text, expect):
    new_text = py.applyToText(changeset, old_text + '\n')
    assert new_text == expect + '\n'


def applyMutations(mu, arrayOfArrays):
    for a in arrayOfArrays:
        method = getattr(mu, a[0])
        result = method(*a[1:])
        if a[0] == 'remove' and a[3]:
            assert a[3] == result


def mutationsToChangeset(oldLen, arrayOfArrays):
    assem = js.smartOpAssembler()
    op = py.Op()
    bank = js.stringAssembler()
    oldPos = 0
    newLen = 0
    for a in arrayOfArrays:
        if a[0] == 'skip':
            op.opcode = '='
            op.chars = a[1]
            op.lines = a[2] if len(a) > 2 else 0
            assem.append(op)
            oldPos += op.chars
            newLen += op.chars
        elif a[0] == 'remove':
            op.opcode = '-'
            op.chars = a[1]
            op.lines = a[2] if len(a) > 2 else 0
            assem.append(op)
            oldPos += op.chars
        elif a[0] == 'insert':
            op.opcode = '+'
            bank.append(a[1])
            op.chars = len(a[1])
            op.lines = a[2] if len(a) > 2 else 0
            assem.append(op)
            newLen += op.chars
    newLen += oldLen - oldPos
    assem.endDocument()
    return py.pack(oldLen, newLen, assem.toString(), bank.toString())


def runMutationTest(testId, origLines, muts, correct):
    print(f'> runMutationTest#{testId}')
    lines = origLines[:]
    mu = py.TextLinesMutator(lines)
    applyMutations(mu, muts)
    mu.close()
    assert lines == correct

    inText = ''.join(origLines)
    cs = mutationsToChangeset(len(inText), muts)
    lines = origLines[:]
    py.mutateTextLines(cs, lines)
    assert lines == correct

    correctText = ''.join(correct)
    outText = py.applyToText(cs, inText)
    assert outText == correctText


def test_mutation():
    runMutationTest(
        1,
        ["apple\n",
         "banana\n",
         "cabbage\n",
         "duffle\n",
         "eggplant\n"],
        [['remove', 1, 0, "a"],
         ['insert', "tu"],
         ['remove', 1, 0, "p"],
         ['skip', 4, 1],
         ['skip', 7, 1],
         ['insert', "cream\npie\n", 2],
         ['skip', 2],
         ['insert', "bot"],
         ['insert', "\n", 1],
         ['insert', "bu"],
         ['skip', 3],
         ['remove', 3, 1, "ge\n"],
         ['remove', 6, 0, "duffle"]],
        ["tuple\n",
         "banana\n",
         "cream\n",
         "pie\n",
         "cabot\n",
         "bubba\n",
         "eggplant\n"])


def mutations_to_ops(mutations) -> Tuple[List[py.Op], str]:
    ops = []
    char_bank = []
    for mutation in mutations:
        if mutation[0] == 'insert':
            text = mutation[1]
            op = py.Op(opcode='+',
                       chars=len(text),
                       lines=text.count('\n'))
            char_bank.append(text)
        elif mutation[0] == 'remove':
            op = py.Op(opcode='-',
                       chars=mutation[1],
                       lines=mutation[2] if len(mutation) >= 3 else 0)
            if len(mutation) >= 4:
                op.check = mutation[3]
        elif mutation[0] == 'skip':
            op = py.Op(opcode='=',
                       chars=mutation[1],
                       lines=mutation[2] if len(mutation) >= 3 else 0)
        else:
            raise ValueError(f'Invalid mutation "{mutation[0]}"')
        ops.append(op)
    return ops, ''.join(char_bank)


@pytest.mark.parametrize(
    'original, mutations, expect_mutated, expect_changes',
    [(["apple\n",
       "banana\n",
       "cabbage\n",
       "duffle\n",
       "eggplant\n"],
      [['remove', 1, 0, "a"],
       ['insert', "tu"],
       ['remove', 1, 0, "p"],
       ['skip', 4, 1],
       ['skip', 7, 1],
       ['insert', "cream\npie\n", 2],
       ['skip', 2],
       ['insert', "bot"],
       ['insert', "\n", 1],
       ['insert', "bu"],
       ['skip', 3],
       ['remove', 3, 1, "ge\n"],
       ['remove', 6, 0, "duffle"]],
      ["tuple\n",
       "banana\n",
       "cream\n",
       "pie\n",
       "cabot\n",
       "bubba\n",
       "eggplant\n"],
      [((0, 1), (0, 1), '-', [1, 0, 'a']),
       ((0, 1), (0, 1), '+', [2, 0, 'tu']),
       ((0, 1), (0, 1), '-', [1, 0, 'p']),
       ((2, 2), (2, 4), '+', [10, 2, 'cream\npie\n']),
       ((2, 3), (4, 5), '+', [3, 0, 'bot']),
       ((2, 3), (4, 5), '+', [1, 1, '\n']),
       ((2, 3), (5, 6), '+', [2, 0, 'bu']),
       ((2, 4), (5, 6), '-', [3, 1, 'ge\n']),
       ((3, 4), (5, 6), '-', [6, 0, 'duffle'])]),

     (["apple\n",
       "banana\n",
       "cabbage\n",
       "duffle\n",
       "eggplant\n"],
      [['remove', 1, 0, "a"],
       ['remove', 1, 0, "p"],
       ['insert', "tu"],
       ['skip', 11, 2],
       ['insert', "cream\npie\n", 2],
       ['skip', 2],
       ['insert', "bot"],
       ['insert', "\n", 1],
       ['insert', "bu"],
       ['skip', 3],
       ['remove', 3, 1, "ge\n"],
       ['remove', 6, 0, "duffle"]],
      ["tuple\n",
       "banana\n",
       "cream\n",
       "pie\n",
       "cabot\n",
       "bubba\n",
       "eggplant\n"],
      [((0, 1), (0, 1), '-', [1, 0, 'a']),
       ((0, 1), (0, 1), '-', [1, 0, 'p']),
       ((0, 1), (0, 1), '+', [2, 0, 'tu']),
       ((2, 2), (2, 4), '+', [10, 2, 'cream\npie\n']),
       ((2, 3), (4, 5), '+', [3, 0, 'bot']),
       ((2, 3), (4, 5), '+', [1, 1, '\n']),
       ((2, 3), (5, 6), '+', [2, 0, 'bu']),
       ((2, 4), (5, 6), '-', [3, 1, 'ge\n']),
       ((3, 4), (5, 6), '-', [6, 0, 'duffle'])]),

     (["apple\n",
       "banana\n",
       "cabbage\n",
       "duffle\n",
       "eggplant\n"],
      [['remove', 6, 1, "apple\n"],
       ['skip', 15, 2],
       ['skip', 6],
       ['remove', 1, 1, "\n"],
       ['remove', 8, 0, "eggplant"],
       ['skip', 1, 1]],
      ["banana\n",
       "cabbage\n",
       "duffle\n"],
      [((0, 2), (0, 1), '-', [6, 1, 'apple\n']),
       ((3, 5), (2, 3), '-', [1, 1, '\n']),
       ((4, 5), (2, 3), '-', [8, 0, 'eggplant'])]),

     (["15\n"],
      [['skip', 1],
       ['insert', "\n2\n3\n4\n", 4],
       ['skip', 2, 1]],
      ["1\n",
       "2\n",
       "3\n",
       "4\n",
       "5\n"],
      # TODO    (0, 5) since '15' was split?
      [((0, 1), (0, 4), '+', [7, 4, '\n2\n3\n4\n'])]),

     (["1\n",
       "2\n",
       "3\n",
       "4\n",
       "5\n"],
      [['skip', 1],
       ['remove', 7, 4, "\n2\n3\n4\n"],
       ['skip', 2, 1]],
      ["15\n"],
      [((0, 5), (0, 1), '-', [7, 4, '\n2\n3\n4\n'])]),

     (["123\n",
       "abc\n",
       "def\n",
       "ghi\n",
       "xyz\n"],
      [['insert', "0"],
       ['skip', 4, 1],
       ['skip', 4, 1],
       ['remove', 8, 2, "def\nghi\n"],
       ['skip', 4, 1]],
      ["0123\n",
       "abc\n",
       "xyz\n"],
      [((0, 1), (0, 1), '+', [1, 0, '0']),
       #(2, 4), (2, 2) instead?   TODO:
       ((2, 5), (2, 3), '-', [8, 2, 'def\nghi\n'])]),

     (["apple\n",
       "banana\n",
       "cabbage\n",
       "duffle\n",
       "eggplant\n"],
      [['remove', 6, 1, "apple\n"],
       ['skip', 15, 2, True],
       ['skip', 6, 0, True],
       ['remove', 1, 1, "\n"],
       ['remove', 8, 0, "eggplant"],
       ['skip', 1, 1, True]],
      ["banana\n",
       "cabbage\n",
       "duffle\n"],
      [((0, 2), (0, 1), '-', [6, 1, 'apple\n']),
       ((3, 5), (2, 3), '-', [1, 1, '\n']),
       ((4, 5), (2, 3), '-', [8, 0, 'eggplant'])])])
def test_mutation(original, mutations, expect_mutated, expect_changes):
    print(f'Original: {original}')
    print(f'Mutations: {mutations}')
    print(f'Expect: {expect_mutated}')
    ops, char_bank = mutations_to_ops(mutations)
    result, changes = py.mutate(original, ops, char_bank)
    print(f'Got: {result}')
    assert result == expect_mutated
    assert changes == expect_changes
