import re

import numpy as np


NUMBER = r"([0-9]+(?:\.[0-9]+)?)"

PT_STAGE_MAP = {
    "pt2": 2.0,
    "pt2a": 2.1,
    "pt2b": 2.2,
    "pt2c": 2.3,
    "pt3a": 3.1,
    "pt3b": 3.2,
    "pt4": 4.0,
    "pt4a": 4.1,
    "pt4b": 4.2,
}


def search_float(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.I)

    if not m:
        return np.nan

    try:
        return float(m.group(1).rstrip(".,;:"))
    except Exception:
        print("Failed float parse")
        print("Pattern:", pattern)
        print("Match:", repr(m.group(1)))
        raise


def search_int(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.I)

    if not m:
        return np.nan

    return int(m.group(1))


def has_phrase(text: str, phrase: str) -> int:
    return int(phrase.lower() in text.lower())


def extract_biopsy_sessions(text: str):
    text = text.lower()

    if "underwent one biopsy session" in text:
        return 1

    m = re.search(
        r"underwent\s+(\d+)\s+biopsy sessions",
        text,
        flags=re.I,
    )

    if m:
        return int(m.group(1))

    return np.nan


def extract_first_biopsy_gleason(text: str):
    m = re.search(
        r"at biopsy,\s*gleason\s+(\d)\+(\d).*?isup grade group\s+(\d)",
        text,
        flags=re.I | re.S,
    )

    if not m:
        return np.nan, np.nan, np.nan

    return (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
    )


def extract_max_biopsy_isup(text: str):
    matches = re.findall(
        r"isup grade group\s+(\d)",
        text,
        flags=re.I,
    )

    if not matches:
        return np.nan

    return max(int(x) for x in matches)


def extract_biopsy_ai_isup(text: str):
    matches = re.findall(
        r"ai model-predicted isup\s+(\d)",
        text,
        flags=re.I,
    )

    if not matches:
        return np.nan

    return max(int(x) for x in matches)


def extract_rp_gleason(text: str):
    m = re.search(
        r"robot-assisted radical prostatectomy specimen showed\s+"
        r"gleason\s+(\d)\+(\d)"
        r"(?:\s+with tertiary pattern\s+(\d))?"
        r".*?isup grade group\s+(\d)",
        text,
        flags=re.I | re.S,
    )

    if not m:
        return np.nan, np.nan, np.nan, np.nan

    return (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)) if m.group(3) else 0,
        int(m.group(4)),
    )


def extract_pt_stage(text: str):
    m = re.search(
        r"pathological stage\s+(pT[0-9][a-z]?)",
        text,
        flags=re.I,
    )

    if not m:
        return None, np.nan

    stage = m.group(1).lower()

    return stage, PT_STAGE_MAP.get(stage, np.nan)


# def extract_node_status(text: str):
#     t = text.lower()

#     if "lymph node metastasis was present" in t:
#         return 1, 1

#     if "there was no lymph node metastasis" in t:
#         return 0, 1

#     if "no lymph nodes were removed" in t:
#         return np.nan, 0

#     return np.nan, np.nan

def extract_node_status(text: str):
    t = text.lower()

    if "lymph node metastasis was present" in t:
        return 1, 1

    if "there was no lymph node metastasis" in t:
        return 0, 1

    if "no lymph nodes were removed" in t:
        return -1, 0

    return np.nan, np.nan


def extract_pirads_max(text: str):
    """
    Handles both:

    PI-RADS: 5
    PI-RADS: 2, 4
    """

    m = re.search(
        r"pi-rads:\s*([0-9,\s]+)",
        text,
        flags=re.I,
    )

    if not m:
        return np.nan

    values = [
        int(v)
        for v in re.findall(r"\d+", m.group(1))
    ]

    return max(values) if values else np.nan


def extract_features(clinical_data: dict):
    radiology = clinical_data.get("radiology_report", "")
    pathology = clinical_data.get("pathology_report", "")
    surgery = clinical_data.get("surgical_pathology_report", "")

    features = {}

    #
    # Radiology
    #

    features["prostate_volume"] = search_float(
        rf"prostate volume:\s*{NUMBER}",
        radiology,
    )

    features["psad"] = search_float(
        rf"psa density:\s*{NUMBER}",
        radiology,
    )

    features["ai_cspca_prob"] = search_float(
        rf"probability of clinically significant prostate cancer.*?:\s*{NUMBER}",
        radiology,
    )

    features["pirads_max"] = extract_pirads_max(radiology)

    #
    # Biopsy
    #

    features["n_biopsy_sessions"] = extract_biopsy_sessions(
        pathology
    )

    (
        features["bx_gl_prim"],
        features["bx_gl_sec"],
        features["bx_isup_first"],
    ) = extract_first_biopsy_gleason(pathology)

    features["bx_isup_max"] = extract_max_biopsy_isup(
        pathology
    )

    features["bx_ai_isup_max"] = extract_biopsy_ai_isup(
        pathology
    )

    features["bx_missing"] = int(
        "gleason pattern and isup report missing"
        in pathology.lower()
    )

    features["bx_cribriform"] = int(
        "cribriform pattern was present"
        in pathology.lower()
    )

    features["bx_intraductal"] = int(
        "intraductal carcinoma was present"
        in pathology.lower()
        or "intraductal carcinoma and"
        in pathology.lower()
    )

    features["bx_perineural"] = int(
        "perineural invasion was present"
        in pathology.lower()
    )

    #
    # Surgery
    #

    (
        features["rp_gl_prim"],
        features["rp_gl_sec"],
        features["rp_tertiary"],
        features["rp_isup"],
    ) = extract_rp_gleason(surgery)

    (
        features["pt_stage"],
        features["pt_stage_num"],
    ) = extract_pt_stage(surgery)

    s = surgery.lower()

    features["epe"] = int(
        "extraprostatic extension was present" in s
    )

    features["margin_positive"] = int(
        "surgical margins were positive" in s
    )

    features["svi"] = int(
        "seminal vesicles were invaded" in s
    )

    features["lvi"] = int(
        "lymphovascular invasion was present" in s
    )

    (
        features["node_positive"],
        features["node_sampled"],
    ) = extract_node_status(surgery)

    #
    # Derived
    #

    if (
        not np.isnan(features["rp_isup"])
        and not np.isnan(features["bx_isup_max"])
    ):
        features["isup_upgrade"] = (
            features["rp_isup"]
            - features["bx_isup_max"]
        )
    else:
        features["isup_upgrade"] = np.nan

    features["high_risk_count"] = (
        features["epe"]
        + features["margin_positive"]
        + features["svi"]
        + features["lvi"]
        + int(features["node_positive"] == 1)
    )

    return features