import js2py.pyjs


def eval_js_module(path, **modules):

    def require(js_path):
        for module_name, module in modules.items():
            if js_path.to_python().endswith(f'/{module_name}'):
                return module

    exports = js2py.pyjs.Scope({})
    context = js2py.EvalJs({'require': require,
                            'exports': exports,
                            'module': {'exports': exports}})
    js2py.run_file(path, context)
    return context