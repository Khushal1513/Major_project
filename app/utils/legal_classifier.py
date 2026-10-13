"""legal_classifier.py – rule-based scanner for Indian constitutional references."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

@dataclass
class LegalClassification:
    constitutional_articles: list[str] = field(default_factory=list)
    fundamental_rights: list[str] = field(default_factory=list)
    directive_principles: list[str] = field(default_factory=list)
    statutes: list[str] = field(default_factory=list)
    is_legal_document: bool = False

FR_ARTICLES = {
    "12": "Definition of State", "13": "Laws inconsistent with FRs",
    "14": "Right to Equality", "15": "Prohibition of Discrimination",
    "16": "Equality of Opportunity", "17": "Abolition of Untouchability",
    "18": "Abolition of Titles", "19": "Right to Freedom",
    "20": "Protection against Arbitrary Conviction",
    "21": "Right to Life and Personal Liberty", "21A": "Right to Education",
    "22": "Protection against Arbitrary Arrest", "23": "Prohibition of Trafficking",
    "24": "Prohibition of Child Labour", "25": "Freedom of Religion",
    "26": "Freedom to Manage Religious Affairs", "29": "Cultural Rights",
    "30": "Rights of Minorities", "32": "Right to Constitutional Remedies",
}
DPSP_ARTICLES = {
    "38": "Social Order", "39": "Certain Policy Principles", "39A": "Equal Justice",
    "40": "Village Panchayats", "41": "Right to Work", "42": "Maternity Relief",
    "43": "Living Wage", "44": "Uniform Civil Code", "45": "Early Childhood Care",
    "46": "Education of SC/ST", "47": "Nutrition & Health", "48A": "Protection of Environment",
    "50": "Separation of Judiciary", "51": "International Peace",
}
STATUTES = [
    (re.compile(r"\bI\.?P\.?C\.?\b|Indian Penal Code", re.I), "Indian Penal Code (IPC)"),
    (re.compile(r"\bC\.?r\.?P\.?C\.?\b|Code of Criminal Procedure", re.I), "CrPC"),
    (re.compile(r"\bBNS\b|Bharatiya Nyaya Sanhita", re.I), "Bharatiya Nyaya Sanhita (BNS)"),
    (re.compile(r"\bBNSS\b|Bharatiya Nagarik Suraksha Sanhita", re.I), "BNSS"),
    (re.compile(r"\bBSA\b|Bharatiya Sakshya Adhiniyam", re.I), "Bharatiya Sakshya Adhiniyam (BSA)"),
    (re.compile(r"\bC\.?P\.?C\.?\b|Code of Civil Procedure", re.I), "CPC"),
    (re.compile(r"\bRTI\b|Right to Information Act", re.I), "RTI Act"),
    (re.compile(r"\bPOCSO\b", re.I), "POCSO Act"),
    (re.compile(r"\bIT Act\b|Information Technology Act", re.I), "IT Act"),
    (re.compile(r"\bUAPA\b|Unlawful Activities", re.I), "UAPA"),
    (re.compile(r"\bNSA\b|National Security Act", re.I), "NSA"),
    (re.compile(r"\bMotor Vehicles Act\b", re.I), "Motor Vehicles Act"),
    (re.compile(r"\bConsumer Protection Act\b", re.I), "Consumer Protection Act"),
]
LEGAL_KW = [
    "whereas", "hereinafter", "notwithstanding", "pursuant to",
    "sub-section", "section", "clause", "article", "schedule",
    "notification", "gazette", "ordinance", "supreme court", "high court",
    "writ", "habeas corpus", "fundamental right", "fir", "bail",
    "cognizable", "affidavit", "decree", "judgment",
]
_ART = re.compile(r"\b(?:Article|Art\.?)\s*(\d{1,3}[A-Z]?)\b", re.I)


class LegalClassifier:
    def classify(self, text: str) -> LegalClassification:
        result = LegalClassification()
        mentioned = set(_ART.findall(text))
        for num in sorted(mentioned, key=self._key):
            result.constitutional_articles.append(f"Article {num}")
            if num in FR_ARTICLES:
                result.fundamental_rights.append(f"Article {num} – {FR_ARTICLES[num]}")
            elif num in DPSP_ARTICLES:
                result.directive_principles.append(f"Article {num} – {DPSP_ARTICLES[num]}")
        for pat, label in STATUTES:
            if pat.search(text):
                result.statutes.append(label)
        kw_hits = sum(1 for kw in LEGAL_KW if kw in text.lower())
        result.is_legal_document = kw_hits >= 3 or bool(result.constitutional_articles) or bool(result.statutes)
        return result

    @staticmethod
    def _key(s: str):
        m = re.match(r"(\d+)([A-Z]?)", s)
        return (int(m.group(1)), m.group(2)) if m else (9999, "")