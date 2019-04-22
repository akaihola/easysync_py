from changeset import changeset
from js_module import eval_js_module

attributepool = eval_js_module('AttributePool.js').module.exports
easysync_tests = eval_js_module('easysync_tests.js',
                                Changeset=changeset,
                                AttributePool=attributepool)
