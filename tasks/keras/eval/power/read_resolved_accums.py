"""
read_resolved_accums.py

Post-convert hook: after hls4ml.converters.convert_from_keras_model has run,
InferPrecisionTypes has already resolved every 'auto' precision (including the
accumulators we set to 'auto' in sweep_hls.py). This reads the *resolved*
accum (and result) widths back off the built hls_model so they can be stored
in meta — letting downstream analysis correlate power/resources against the
precision actually synthesized, not the 'auto' we requested.

Call it AFTER convert_from_keras_model returns and BEFORE/AFTER write()/build()
(the types are fixed once convert returns). Example, in build_hls_project:

    hls_model = hls4ml.converters.convert_from_keras_model(...)
    resolved = read_resolved_accums(hls_model)
    config["meta"]["resolved_precision"] = resolved   # persist into the doc
    log.info(f"[{uuid}] resolved accums: "
             + ", ".join(f"{n}:{v['accum']}" for n, v in resolved.items()))
    hls_model.write()
    ...

Robustness note: the accessor for a layer's accumulator type has shifted
slightly across hls4ml versions. We try, in order:
  1. layer.get_attr('accum_t')            (NamedType)         -> .precision
  2. layer.attributes['accum_t']          (dict-like)
  3. scan layer.types / layer.attributes for any key containing 'accum'
and stringify whatever precision object we find. FixedPrecisionType stringifies
to 'ap_fixed<W,I>' (or 'ap_ufixed<...>'). On any failure we record None for
that layer rather than raising, so a read-back hiccup never kills a build.
The first time you run this, eyeball the logged widths to confirm the accessor
is hitting on your installed version.
"""
import re


_MAC_CLASS_HINTS = ("dense", "conv")


def _precision_to_str(prec_obj):
    """Best-effort stringify of an hls4ml precision/NamedType object.

    FixedPrecisionType.__str__ already yields 'ap_fixed<W,I>' (signed) or
    'ap_ufixed<W,I>' (unsigned). If we were handed a NamedType, descend to its
    .precision first. As a last resort, build the string from width/integer/
    signed attributes."""
    if prec_obj is None:
        return None

    # NamedType -> unwrap to its .precision
    inner = getattr(prec_obj, "precision", prec_obj)

    s = str(inner)
    if "fixed<" in s:
        # normalize whitespace
        return s.replace(" ", "")

    # Fall back to constructing from numeric attributes.
    width = getattr(inner, "width", None)
    integer = getattr(inner, "integer", None)
    signed = getattr(inner, "signed", True)
    if width is not None and integer is not None:
        kind = "ap_fixed" if signed else "ap_ufixed"
        return f"{kind}<{int(width)},{int(integer)}>"
    return None


def _get_named_type(layer, type_name):
    """Try several access paths for a layer's named HLS type (e.g. 'accum_t')."""
    # 1. get_attr
    getter = getattr(layer, "get_attr", None)
    if callable(getter):
        try:
            t = getter(type_name)
            if t is not None:
                return t
        except Exception:
            pass

    # 2. attributes dict-like
    attrs = getattr(layer, "attributes", None)
    if attrs is not None:
        try:
            if type_name in attrs:
                return attrs[type_name]
        except Exception:
            pass

    # 3. a .types mapping, if present
    types = getattr(layer, "types", None)
    if isinstance(types, dict) and type_name in types:
        return types[type_name]

    return None


def _scan_for_accum(layer):
    """Last-ditch: scan attributes/types for any key containing 'accum'."""
    for container_name in ("attributes", "types"):
        container = getattr(layer, container_name, None)
        if container is None:
            continue
        try:
            items = container.items()
        except Exception:
            continue
        for key, val in items:
            if isinstance(key, str) and "accum" in key.lower():
                s = _precision_to_str(val)
                if s:
                    return s
    return None


def _is_mac_layer(layer):
    cls = type(layer).__name__.lower()
    return any(h in cls for h in _MAC_CLASS_HINTS)


def read_resolved_accums(hls_model, mac_only=True):
    """Return {layer_name: {'accum': '<str|None>', 'result': '<str|None>'}}.

    mac_only: if True (default), only report Dense/Conv layers — the ones with
    an accumulator worth tracking. Set False to dump every layer.
    """
    out = {}
    try:
        layers = hls_model.get_layers()
    except Exception:
        return out

    for layer in layers:
        name = getattr(layer, "name", None) or "<unknown>"
        if mac_only and not _is_mac_layer(layer):
            continue

        # accum
        accum_t = _get_named_type(layer, "accum_t")
        accum_str = _precision_to_str(accum_t) if accum_t is not None else None
        if accum_str is None:
            accum_str = _scan_for_accum(layer)

        # result (output variable precision)
        result_str = None
        try:
            ov = layer.get_output_variable()
            if ov is not None:
                result_str = _precision_to_str(getattr(ov, "type", ov))
        except Exception:
            pass

        out[name] = {"accum": accum_str, "result": result_str}

    return out
