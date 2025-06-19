import spacy
import networkx as nx

nlp = spacy.load("en_core_web_sm")


def extract_concept_map(text):
    doc = nlp(text)
    G = nx.DiGraph()

    for sent in doc.sents:
        for chunk in sent.noun_chunks:
            G.add_node(chunk.text.strip())

        for token in sent:
            if token.dep_ in ("nsubj", "dobj", "attr", "prep"):
                subj = token.head.text.strip()
                obj = token.text.strip()
                if subj != obj:
                    G.add_edge(subj, obj, label=token.dep_)

    return G
