"""Test helpers and a tiny fictional multi-provider corpus."""

import numpy as np

PROVIDER_DOCS: dict[int, list[str]] = {
    0: [
        "The Riverside council approved a new budget for the large park on Monday after a long debate. "
        "Mayor Elena Brooks said the plan will help many families and also create jobs for local workers. "
        "Critics argued that the cost was too high and that the construction timeline was unrealistic. "
        "The park will include a community garden, a playground and a small open-air theatre.",
        "Residents of Riverside gathered at the town hall to discuss the future of the old railway station. "
        "Several speakers proposed turning the station into a museum about the history of the region. "
        "The heritage society said the building is important and should be protected from demolition. "
        "A final decision is expected after an engineering survey is completed in the spring.",
    ],
    1: [
        "Chef Marco Lin opened a restaurant in the harbour district that serves dishes made from local seafood. "
        "His signature recipe combines smoked mackerel with pickled fennel and a sauce of roasted peppers. "
        "Food critics praised the menu but noted that the prices are high for a neighbourhood restaurant. "
        "Lin said he buys fish every morning directly from the boats that dock near the old lighthouse.",
        "The annual harbour food festival attracted thousands of visitors despite the heavy rain on Saturday. "
        "Vendors sold grilled oysters, fish soup and sweet pastries filled with almond cream. "
        "Organisers said the festival raised money for the maritime rescue volunteers of the coast. "
        "Next year the event will move to a larger square near the ferry terminal.",
    ],
    2: [
        "The regional hospital introduced a new triage system to reduce waiting times in the emergency ward. "
        "Doctors use a digital dashboard that ranks patients by the severity of their symptoms. "
        "Nurses reported that the system shows clear benefits during the busy winter season. "
        "The health board plans to evaluate the results and extend the system to two other hospitals.",
        "A local school launched a programme that teaches students how to grow vegetables on the roof. "
        "Teachers said the project helps children understand nutrition, biology and teamwork. "
        "Parents donated seeds, soil and wooden boxes to build the first raised beds last autumn. "
        "The harvest is used in the school kitchen and the surplus is given to a nearby food bank.",
    ],
}


def rotate(keys: list[str]) -> list[str]:
    """Wrong-key control used by the paper runner: provider i gets key i+1."""
    return keys[1:] + keys[:1]


def unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


_WORDS = (
    "the a and but also of to in on for with from about many large big important help used said shows "
    "council budget park river station museum harbour market festival hospital school garden bridge library "
    "mayor chef doctor teacher nurse farmer artist engineer resident visitor student volunteer officer "
    "approved opened built repaired announced described praised criticised funded planned delayed launched "
    "new old small quiet busy local regional annual public private historic modern rural coastal northern "
    "morning evening winter spring summer autumn week month year decade season weekend holiday "
    "budget report survey project programme plan proposal contract scheme network service system "
    "village town district valley island coast hill forest lake road square tower factory"
).split()


def synthetic_corpus(n_providers: int, docs_per_provider: int, sents_per_doc: int, seed: int = 0) -> dict[int, list[str]]:
    """Random-word documents with distinct sentences (for statistically independent windows)."""
    import random

    rng = random.Random(seed)
    corpus: dict[int, list[str]] = {}
    for p in range(n_providers):
        docs = []
        for _ in range(docs_per_provider):
            sents = []
            for _ in range(sents_per_doc):
                words = [rng.choice(_WORDS) for _ in range(rng.randint(10, 40))]
                sents.append(" ".join(words).capitalize() + ".")
            docs.append(" ".join(sents))
        corpus[p] = docs
    return corpus
