def build_object_from_class_name(cls_str, parent_cls, strict, *args, **kwargs):
    import importlib

    pkg = ".".join(cls_str.split(".")[:-1])
    cls_name = cls_str.split(".")[-1]
    cls_type = getattr(importlib.import_module(pkg), cls_name)
    if parent_cls is not None:
        assert issubclass(cls_type, parent_cls), f'| {cls_type} is not subclass of {parent_cls}.'
    if strict:
        return cls_type(*args, **kwargs)
    return cls_type(*args, **filter_kwargs(kwargs, cls_type))


def filter_kwargs(dict_to_filter, kwarg_obj):
    import inspect

    sig = inspect.signature(kwarg_obj)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        # the signature contains definitions like **kwargs, so there is no need to filter
        return dict_to_filter.copy()
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD or param.kind == param.KEYWORD_ONLY
    ]
    filtered_dict = {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys if
                     filter_key in dict_to_filter}
    return filtered_dict
