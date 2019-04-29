from pathlib import Path

from pkg_resources import resource_filename

from easysync_py.changeset import changeset
from easysync_py.js_module import eval_js_module

ATTRIBUTEPOOL_PATH = resource_filename('easysync_py', 'AttributePool.js')
attributepool = eval_js_module(ATTRIBUTEPOOL_PATH).module.exports

TESTS_PATH = Path(__file__).parent / 'easysync_tests.js'
easysync_tests = eval_js_module(TESTS_PATH,
                                Changeset=changeset,
                                AttributePool=attributepool)
