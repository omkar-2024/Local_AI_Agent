import json
import math
import re
from typing import Any


# ============================================================
# ENGINEERING CALCULATOR
# Deterministic, local-only calculations. No LLM inference.
# ============================================================


def _normalise_text(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "³": "^3",
        "²": "^2",
        "°": "",
        "Δ": "delta",
        "μ": "mu",
        "η": "eta",
        "ρ": "rho",
        "ṁ": "mass flow",
        "×": "*",
        "–": "-",
        "—": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.lower().replace("\u00a0", " ")


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.")

    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def _positive(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _fraction(value: Any, name: str) -> float:
    result = _number(value, name)
    if result > 1:
        result /= 100.0
    if result <= 0 or result > 1:
        raise ValueError(f"{name} must be between 0 and 1 or 0% and 100%.")
    return result


def _parse_value(value: Any, default_unit: str = "") -> tuple[float, str]:
    if isinstance(value, (int, float)):
        return float(value), default_unit

    text = _normalise_text(value).strip()
    match = re.search(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"\s*([a-zA-Z0-9.^/*_%\-]+)?",
        text,
    )
    if not match:
        raise ValueError(f"Could not parse value '{value}'.")

    return float(match.group(1)), (match.group(2) or default_unit).lower()


def _flow_m3s(value: Any) -> float:
    number, unit = _parse_value(value, "m3/s")
    unit = unit.replace(" ", "")

    if unit in ("m3/s", "m^3/s", "m3sec", "m^3sec", ""):
        return number
    if unit in ("m3/h", "m^3/h", "m3/hr", "m^3/hr"):
        return number / 3600.0
    if unit in ("l/s", "l/sec", "litre/s", "liter/s"):
        return number / 1000.0
    if unit in ("l/min", "l/minute", "litre/min", "liter/min"):
        return number / 60000.0
    if unit in ("l/h", "l/hr"):
        return number / 3_600_000.0
    raise ValueError(f"Unsupported flow unit: {unit}")


def _length_m(value: Any) -> float:
    number, unit = _parse_value(value, "m")
    unit = unit.replace(" ", "")

    if unit in ("m", "meter", "meters", "metre", "metres", ""):
        return number
    if unit in ("mm", "millimeter", "millimeters", "millimetre", "millimetres"):
        return number / 1000.0
    if unit in ("cm", "centimeter", "centimeters", "centimetre", "centimetres"):
        return number / 100.0
    if unit in ("km", "kilometer", "kilometers", "kilometre", "kilometres"):
        return number * 1000.0
    if unit in ("ft", "foot", "feet"):
        return number * 0.3048
    if unit in ("in", "inch", "inches"):
        return number * 0.0254
    raise ValueError(f"Unsupported length unit: {unit}")


def _area_m2(value: Any) -> float:
    number, unit = _parse_value(value, "m2")
    unit = unit.replace(" ", "")

    if unit in ("m2", "m^2", "sqm", ""):
        return number
    if unit in ("cm2", "cm^2"):
        return number / 10_000.0
    if unit in ("mm2", "mm^2"):
        return number / 1_000_000.0
    if unit in ("ft2", "ft^2", "sqft"):
        return number * 0.09290304
    if unit in ("in2", "in^2", "sqin"):
        return number * 0.00064516
    raise ValueError(f"Unsupported area unit: {unit}")


def _velocity_ms(value: Any) -> float:
    number, unit = _parse_value(value, "m/s")
    unit = unit.replace(" ", "")
    if unit in ("m/s", "m/sec", "ms-1", ""):
        return number
    if unit in ("km/h", "km/hr"):
        return number / 3.6
    if unit in ("ft/s", "ft/sec"):
        return number * 0.3048
    raise ValueError(f"Unsupported velocity unit: {unit}")


def _viscosity_pas(value: Any) -> float:
    number, unit = _parse_value(value, "pa.s")
    unit = unit.replace(" ", "")
    if unit in ("pa.s", "pa*s", "pas", ""):
        return _positive(number, "viscosity")
    if unit in ("cp", "centipoise"):
        return _positive(number / 1000.0, "viscosity")
    if unit in ("mpa.s", "mpas"):
        return _positive(number / 1000.0, "viscosity")
    raise ValueError(f"Unsupported viscosity unit: {unit}")


def _mass_flow_kgs(value: Any) -> float:
    number, unit = _parse_value(value, "kg/s")
    unit = unit.replace(" ", "")
    if unit in ("kg/s", "kg/sec", ""):
        return number
    if unit in ("kg/h", "kg/hr"):
        return number / 3600.0
    if unit in ("g/s", "g/sec"):
        return number / 1000.0
    raise ValueError(f"Unsupported mass-flow unit: {unit}")


def _temperature_difference_k(value: Any) -> float:
    number, unit = _parse_value(value, "k")
    unit = unit.lower()
    if unit in ("k", "c", "", "degc"):
        return number
    if unit in ("f", "degf"):
        return number * 5.0 / 9.0
    raise ValueError(f"Unsupported temperature unit: {unit}")


def _density_kg_m3(value: Any) -> float:
    number, unit = _parse_value(value, "kg/m3")
    unit = unit.replace(" ", "")
    if unit in ("kg/m3", "kg/m^3", ""):
        return _positive(number, "density")
    if unit in ("g/cm3", "g/cm^3"):
        return _positive(number * 1000.0, "density")
    raise ValueError(f"Unsupported density unit: {unit}")


def _pressure_pa(value: Any) -> float:
    number, unit = _parse_value(value, "pa")
    unit = unit.replace(" ", "")
    if unit in ("pa", ""):
        return number
    if unit in ("kpa",):
        return number * 1000.0
    if unit in ("mpa",):
        return number * 1_000_000.0
    if unit in ("bar",):
        return number * 100_000.0
    if unit in ("psi",):
        return number * 6894.757293168
    raise ValueError(f"Unsupported pressure unit: {unit}")


def _modulus_pa(value: Any) -> float:
    number, unit = _parse_value(value, "pa")
    unit = unit.replace(" ", "")
    if unit in ("pa", ""):
        return number
    if unit in ("kpa",):
        return number * 1_000.0
    if unit in ("mpa",):
        return number * 1_000_000.0
    if unit in ("gpa",):
        return number * 1_000_000_000.0
    if unit in ("psi",):
        return number * 6894.757293168
    raise ValueError(f"Unsupported modulus unit: {unit}")


def _conductivity_w_mk(value: Any) -> float:
    number, unit = _parse_value(value, "w/m-k")
    unit = unit.replace(" ", "").replace("°", "")
    if unit in ("w/m-k", "w/mk", "w/m*k", "w/m.k", "w/m-kelvin", ""):
        return _positive(number, "thermal conductivity")
    if unit in ("kw/m-k", "kw/mk", "kw/m*k"):
        return _positive(number * 1000.0, "thermal conductivity")
    raise ValueError(f"Unsupported thermal-conductivity unit: {unit}")


def _udl_n_per_m(value: Any) -> float:
    number, unit = _parse_value(value, "n/m")
    unit = unit.replace(" ", "")
    if unit in ("n/m", "n/meter", "n/metre", ""):
        return number
    if unit in ("kn/m", "kn/meter", "kn/metre"):
        return number * 1000.0
    if unit in ("n/mm",):
        return number * 1000.0
    if unit in ("lbf/ft",):
        return number * 14.59390294
    raise ValueError(f"Unsupported distributed-load unit: {unit}")


def _stress_mpa(value_pa: float) -> float:
    return value_pa / 1_000_000.0


def _natural_parameters(text: str) -> dict[str, Any]:
    text = _normalise_text(text)
    result: dict[str, Any] = {}

    patterns = {
        "flow_rate": r"(?:flow(?:\s+rate)?|discharge)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(m3\s*/\s*h|m\^3\s*/\s*h|m3\s*/\s*s|m\^3\s*/\s*s|l\s*/\s*s|l\s*/\s*min|lpm)",
        "velocity": r"(?:velocity|speed)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(m\s*/\s*s|m\s*/\s*sec|km\s*/\s*h|ft\s*/\s*s)",
        "diameter": r"(?:internal\s+diameter|pipe\s+diameter|diameter|dia\.?)(?:\s+of|\s+is|\s*=|\s*:)?\s*([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches|ft)",
        "length": r"(?:pipe\s+length|length|pipe\s+span)(?:\s+of|\s+is|\s*=|\s*:)?\s*([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m|km|ft|in|inch|inches)",
        "span": r"(?:span|beam\s+span)(?:\s+of|\s+is|\s*=|\s*:)?\s*([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m|ft|in|inch|inches)",
        "density": r"(?:density|rho)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(kg\s*/\s*m3|kg\s*/\s*m\^3)",
        "friction_factor": r"(?:darcy\s+)?(?:friction\s+factor|friction\s+coefficient)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)",
        "head": r"(?:pump\s+)?head\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(m|ft)",
        "efficiency": r"(?:efficiency|eta)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*%?",
        "viscosity": r"(?:dynamic\s+)?(?:viscosity|mu)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(pa\.?s|cp|centipoise|mpa\.?s)",
        "mass_flow": r"(?:mass\s+flow(?:\s+rate)?)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(kg\s*/\s*s|kg\s*/\s*h|kg\s*/\s*hr|g\s*/\s*s)",
        "cp": r"(?:cp|specific\s+heat|heat\s+capacity)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(kj\s*/\s*kgk|j\s*/\s*kgk)",
        "delta_t": r"(?:delta\s*t|temperature\s+difference|temperature\s+change|temperature\s+difference\s+of)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(k|c|f)?",
        "area": r"(?:surface\s+area|wall\s+area|area)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(m\^?2|cm\^?2|mm\^?2|ft\^?2|in\^?2|sqm|sqft)?",
        "thickness": r"(?:wall\s+)?thickness\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)",
        "conductivity": r"(?:thermal\s+)?conductivity\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(w\s*/\s*m\s*-?\s*k|w\s*/\s*m\.?k|kw\s*/\s*m\s*-?\s*k)?",
        "width": r"(?:beam\s+)?width\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)",
        "height": r"(?:beam\s+)?height\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)",
        "udl": r"(?:uniform(?:ly)?\s+distributed\s+load|distributed\s+load|udl|uniform\s+load)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(kn\s*/\s*m|n\s*/\s*m|n\s*/\s*mm|lbf\s*/\s*ft)",
        "youngs_modulus": r"(?:young'?s\s+modulus|elastic\s+modulus|modulus\s+of\s+elasticity|e)\s*(?:is|=|of|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(gpa|mpa|kpa|pa|psi)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1)
        unit = ""
        if match.lastindex and match.lastindex >= 2:
            unit = (match.group(2) or "").strip()
        result[key] = f"{value} {unit}".strip()

    # Generic temperature-difference fallback for wording such as
    # "a temperature difference of 40 C".
    if "delta_t" not in result:
        match = re.search(
            r"temperature\s+difference\s+(?:of|is|=)\s*([-+]?\d+(?:\.\d+)?)\s*(k|c|f)?",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            result["delta_t"] = f"{match.group(1)} {(match.group(2) or 'K')}".strip()

    # Flowing-speed wording: "carrying water at 2 m/s" or "flowing at 2 m/s".
    if "velocity" not in result:
        match = re.search(
            r"(?:at|with|velocity\s+of|speed\s+of)\s*([-+]?\d+(?:\.\d+)?)\s*(m\s*/\s*s|m\s*/\s*sec|km\s*/\s*h|ft\s*/\s*s)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            result["velocity"] = f"{match.group(1)} {match.group(2)}"

    # Reverse-order pipe diameter wording: "100 mm diameter pipe".
    if "diameter" not in result:
        match = re.search(
            r"([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m|ft|in|inch|inches)\s+(?:internal\s+)?diameter",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            result["diameter"] = f"{match.group(1)} {match.group(2)}"

    # Reverse-order length wording: "50 m long pipe".
    if "length" not in result:
        match = re.search(
            r"([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m|km|ft|in|inch|inches)\s+(?:long|length)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            result["length"] = f"{match.group(1)} {match.group(2)}"

    # Beam prompts frequently say "a 5 m span" rather than "span is 5 m".
    if "span" not in result:
        match = re.search(
            r"(?:a\s+)?([-+]?\d+(?:\.\d+)?)\s*(m|mm|cm|ft|in|inch|inches)\s+(?:long\s+)?(?:span|beam)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            result["span"] = f"{match.group(1)} {match.group(2)}"

    # Heat-wall prompts often say "0.2 m thick".
    if "thickness" not in result:
        match = re.search(
            r"([-+]?\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)\s+thick",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            result["thickness"] = f"{match.group(1)} {match.group(2)}"

    return result


def _pick(params: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in params and params[name] not in (None, ""):
            return params[name]
    raise KeyError(names[0])


def _calc(name: str, params: dict[str, Any]) -> dict[str, Any]:
    name = _normalise_text(name).strip().replace("-", "_").replace(" ", "_")

    aliases = {
        "pump": "pump_power",
        "power": "pump_power",
        "pressure_loss": "pressure_drop",
        "pressure_drop_friction": "pressure_drop",
        "velocity": "pipe_velocity",
        "flow_velocity": "pipe_velocity",
        "reynolds": "reynolds_number",
        "heat": "heat_duty",
        "wall_heat_transfer": "heat_transfer_wall",
        "flat_wall_heat_transfer": "heat_transfer_wall",
        "heat_transfer": "heat_transfer_wall",
        "beam": "beam_bending",
        "beam_stress": "beam_bending",
        "bending": "beam_bending",
        "bending_stress": "beam_bending",
    }
    name = aliases.get(name, name)
    g = 9.80665

    # ---------------------------------------------------------
    # PIPE VELOCITY
    # ---------------------------------------------------------
    if name == "pipe_velocity":
        q = _positive(_flow_m3s(_pick(params, "flow_rate")), "flow_rate")
        d = _positive(_length_m(_pick(params, "diameter")), "diameter")
        area = math.pi * d**2 / 4.0
        velocity = q / area
        return {
            "calculation": "Pipe flow velocity",
            "formula": "v = Q / A, A = πD²/4",
            "result": velocity,
            "unit": "m/s",
            "inputs": {"flow_rate_m3_s": q, "diameter_m": d},
        }

    # ---------------------------------------------------------
    # REYNOLDS NUMBER
    # ---------------------------------------------------------
    if name == "reynolds_number":
        d = _positive(_length_m(_pick(params, "diameter")), "diameter")
        rho = _density_kg_m3(params.get("density", 1000))
        mu = _viscosity_pas(params.get("viscosity", "1 cP"))
        if "velocity" in params:
            velocity = _positive(_velocity_ms(params["velocity"]), "velocity")
        else:
            q = _positive(_flow_m3s(_pick(params, "flow_rate")), "flow_rate")
            velocity = q / (math.pi * d**2 / 4.0)
        re_number = rho * velocity * d / mu
        regime = "laminar" if re_number < 2300 else ("transitional" if re_number < 4000 else "turbulent")
        return {
            "calculation": "Reynolds number",
            "formula": "Re = ρvD / μ",
            "result": re_number,
            "unit": "dimensionless",
            "flow_regime": regime,
            "inputs": {
                "velocity_m_s": velocity,
                "diameter_m": d,
                "density_kg_m3": rho,
                "viscosity_pa_s": mu,
            },
        }

    # ---------------------------------------------------------
    # PRESSURE DROP - DARCY WEISBACH
    # ΔP = f (L/D) (ρv²/2)
    # ---------------------------------------------------------
    if name == "pressure_drop":
        d = _positive(_length_m(_pick(params, "diameter")), "diameter")
        length = _positive(_length_m(_pick(params, "length")), "length")
        rho = _density_kg_m3(params.get("density", 1000))
        friction_factor = _positive(_number(params["friction_factor"], "friction_factor"), "friction_factor")
        if "velocity" in params:
            velocity = _positive(_velocity_ms(params["velocity"]), "velocity")
            flow_rate = velocity * math.pi * d**2 / 4.0
        else:
            flow_rate = _positive(_flow_m3s(_pick(params, "flow_rate")), "flow_rate")
            velocity = flow_rate / (math.pi * d**2 / 4.0)
        pressure_drop = friction_factor * (length / d) * (rho * velocity**2 / 2.0)
        return {
            "calculation": "Pipe pressure drop",
            "formula": "ΔP = f × (L/D) × (ρv²/2)",
            "result": pressure_drop,
            "unit": "Pa",
            "result_kPa": pressure_drop / 1000.0,
            "result_bar": pressure_drop / 100000.0,
            "inputs": {
                "velocity_m_s": velocity,
                "flow_rate_m3_s": flow_rate,
                "diameter_m": d,
                "length_m": length,
                "density_kg_m3": rho,
                "friction_factor": friction_factor,
            },
        }

    # ---------------------------------------------------------
    # PUMP POWER
    # P = ρgQH/η
    # ---------------------------------------------------------
    if name == "pump_power":
        q = _positive(_flow_m3s(_pick(params, "flow_rate")), "flow_rate")
        head = _positive(_length_m(_pick(params, "head")), "head")
        rho = _density_kg_m3(params.get("density", 1000))
        efficiency = _fraction(params["efficiency"], "efficiency")
        power = rho * g * q * head / efficiency
        return {
            "calculation": "Pump shaft power",
            "formula": "P = ρgQH / η",
            "result": power,
            "unit": "W",
            "result_kW": power / 1000.0,
            "inputs": {
                "flow_rate_m3_s": q,
                "head_m": head,
                "density_kg_m3": rho,
                "efficiency": efficiency,
            },
        }

    # ---------------------------------------------------------
    # HEAT DUTY
    # Q = m_dot Cp ΔT
    # ---------------------------------------------------------
    if name == "heat_duty":
        mass_flow = _positive(_mass_flow_kgs(_pick(params, "mass_flow")), "mass_flow")
        cp_number, cp_unit = _parse_value(_pick(params, "cp"), "J/kgK")
        cp = cp_number * 1000.0 if "kj" in cp_unit.lower().replace(" ", "") else cp_number
        cp = _positive(cp, "cp")
        delta_t = _temperature_difference_k(_pick(params, "delta_t"))
        duty = mass_flow * cp * delta_t
        return {
            "calculation": "Heat duty",
            "formula": "Q = ṁ × Cp × ΔT",
            "result": duty,
            "unit": "W",
            "result_kW": duty / 1000.0,
            "inputs": {"mass_flow_kg_s": mass_flow, "cp_j_kg_k": cp, "delta_t_k": delta_t},
        }

    # ---------------------------------------------------------
    # FLAT-WALL HEAT TRANSFER
    # q = k A ΔT / L
    # ---------------------------------------------------------
    if name == "heat_transfer_wall":
        thickness = _positive(_length_m(_pick(params, "thickness")), "thickness")
        conductivity = _positive(_conductivity_w_mk(_pick(params, "conductivity")), "thermal conductivity")
        area = _positive(_area_m2(_pick(params, "area")), "area")
        delta_t = _positive(_temperature_difference_k(_pick(params, "delta_t")), "temperature difference")
        heat_rate = conductivity * area * delta_t / thickness
        heat_flux = heat_rate / area
        return {
            "calculation": "Heat transfer through a flat wall",
            "formula": "q = k × A × ΔT / L",
            "result": heat_rate,
            "unit": "W",
            "result_kW": heat_rate / 1000.0,
            "heat_flux_W_m2": heat_flux,
            "inputs": {
                "thickness_m": thickness,
                "thermal_conductivity_W_m_K": conductivity,
                "area_m2": area,
                "temperature_difference_K": delta_t,
            },
        }

    # ---------------------------------------------------------
    # SIMPLY SUPPORTED RECTANGULAR BEAM WITH UDL
    # Mmax = wL²/8
    # I = bh³/12
    # σmax = Mc/I
    # δmax = 5wL⁴/(384EI)
    # ---------------------------------------------------------
    if name == "beam_bending":
        span = _positive(_length_m(_pick(params, "span", "length")), "span")
        width = _positive(_length_m(_pick(params, "width")), "width")
        height = _positive(_length_m(_pick(params, "height")), "height")
        udl = _positive(_udl_n_per_m(_pick(params, "udl")), "distributed load")
        youngs_modulus = _modulus_pa(params.get("youngs_modulus", "200 GPa"))

        moment = udl * span**2 / 8.0
        inertia = width * height**3 / 12.0
        c = height / 2.0
        stress = moment * c / inertia
        deflection = 5.0 * udl * span**4 / (384.0 * youngs_modulus * inertia)

        return {
            "calculation": "Simply supported rectangular beam under UDL",
            "formula": "Mmax = wL²/8; I = bh³/12; σmax = Mc/I; δmax = 5wL⁴/(384EI)",
            "result": stress,
            "unit": "Pa",
            "result_MPa": _stress_mpa(stress),
            "maximum_bending_moment_Nm": moment,
            "maximum_deflection_m": deflection,
            "maximum_deflection_mm": deflection * 1000.0,
            "inputs": {
                "span_m": span,
                "width_m": width,
                "height_m": height,
                "udl_N_m": udl,
                "youngs_modulus_Pa": youngs_modulus,
                "moment_of_inertia_m4": inertia,
            },
        }

    raise ValueError(
        "Unsupported calculation. Use pump_power, pressure_drop, "
        "pipe_velocity, reynolds_number, heat_duty, heat_transfer_wall, "
        "beam_bending, or auto."
    )


def _infer_calculation(text: str, params: dict[str, Any]) -> str:
    text = _normalise_text(text)

    if any(x in text for x in ("beam", "bending stress", "bending", "deflection", "simply supported")) or all(
        key in params for key in ("width", "height", "udl")
    ):
        return "beam_bending"

    if any(x in text for x in ("flat wall", "through a wall", "wall that is", "thermal conductivity", "conduction through")) or (
        "thickness" in params and "conductivity" in params and "area" in params
    ):
        return "heat_transfer_wall"

    if any(x in text for x in ("pressure drop", "pressure loss", "darcy", "friction factor")) or "friction_factor" in params:
        return "pressure_drop"

    if any(x in text for x in ("reynolds", "reynolds number")) or "viscosity" in params:
        return "reynolds_number"

    if any(x in text for x in ("pump power", "pump calculation", "pump", "pump head")) or "head" in params:
        return "pump_power"

    if any(x in text for x in ("heat duty", "specific heat", "heat capacity")) or (
        "mass_flow" in params and "cp" in params
    ):
        return "heat_duty"

    if "velocity" in params or "flow velocity" in text:
        return "pipe_velocity" if "flow_rate" in params else "reynolds_number"

    if "flow_rate" in params and "diameter" in params:
        return "pipe_velocity"

    return "pipe_velocity"


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def engineering_calculation(calculation: str, parameters: str) -> str:
    """Run a deterministic local engineering calculation.

    The function accepts JSON parameters or natural-language engineering
    prompts. It deliberately performs arithmetic without an LLM so the
    engineering path is fast, reproducible, and auditable.
    """
    try:
        parameter_text = str(parameters or "").strip()
        if not parameter_text:
            raise ValueError("No engineering parameters were supplied.")

        try:
            parsed = json.loads(parameter_text)
            params = parsed if isinstance(parsed, dict) else {}
        except Exception:
            params = _natural_parameters(parameter_text)

        if not params:
            raise ValueError(
                "No parameters found. Provide values such as diameter, "
                "length, velocity/flow rate, load, thickness, area, or conductivity."
            )

        requested = _normalise_text(calculation).strip()
        calculation_name = _infer_calculation(parameter_text, params) if requested in ("", "auto") else requested
        result = _calc(calculation_name, params)

        lines = [
            "ENGINEERING CALCULATION RESULT",
            f"Calculation: {result['calculation']}",
            f"Formula: {result['formula']}",
            "",
            "Inputs:",
        ]

        for key, value in result.get("inputs", {}).items():
            lines.append(f"- {key}: {_format_value(value)}")

        lines.extend(["", f"Result: {_format_value(result['result'])} {result['unit']}"])

        optional_lines = [
            ("result_kW", "Result (kW)", "kW"),
            ("result_kPa", "Result (kPa)", "kPa"),
            ("result_bar", "Result (bar)", "bar"),
            ("result_MPa", "Maximum bending stress (MPa)", "MPa"),
            ("maximum_bending_moment_Nm", "Maximum bending moment", "N·m"),
            ("maximum_deflection_mm", "Maximum deflection", "mm"),
            ("heat_flux_W_m2", "Heat flux", "W/m²"),
        ]
        for key, label, unit in optional_lines:
            if key in result:
                lines.append(f"{label}: {_format_value(result[key])} {unit}")

        if "flow_regime" in result:
            lines.append(f"Flow regime: {result['flow_regime']}")

        lines.extend([
            "",
            "Calculated deterministically by the local engineering calculator.",
        ])
        return "\n".join(lines)

    except Exception as exc:
        return f"Engineering calculation error: {type(exc).__name__}: {exc}"
