#!/usr/bin/python3

# Copyright 2021 AIR Institute
# See LICENSE for details.

import spacy
import pandas as pd
from typing import List
import xml.etree.ElementTree as ET


class Node:
    """A BPNM node"""

    def __init__(self, n_type, n_id, n_name):
        self.type = n_type
        self.id = n_id
        self.name = n_name

    def __str__(self):
        msg = "------------------ NODO ------------------\n"
        msg += f"ID={self.id}\n"
        msg += f"TYPE={self.type}\n"
        msg += f"NAME={self.name}\n"
        msg += "------------------------------------------\n"
        return msg


class StartEvent(Node):
    def __init__(self, n_id, n_name):
        super().__init__("startEvent", n_id, n_name)


class Task(Node):
    def __init__(self, n_id, n_name):
        super().__init__("task", n_id, n_name)


class Gateway(Node):
    def __init__(self, n_id, n_name):
        super().__init__("gateway", n_id, n_name)


class SequenceFlow(Node):
    def __init__(self, n_id, n_name):
        super().__init__("sequenceFlow", n_id, n_name)


class Parser:
    def __init__(self, path):
        self.items = []
        self.relationships = {}
        self.output = []

        self.__search_process(path)
        for item in self.items:
            if item.type == "startEvent":
                self.__build_text(item, 0)
                break

    def __str__(self):
        msg = "\n------------------ PROCESS RELS ------------------\n"
        for rel in self.relationships:
            msg += f"{rel} -> ["
            for dst in self.relationships[rel]:
                msg += f"{dst}, "
            msg += "]\n"
        msg += "---------------------------------------------------\n"
        return msg

    def __search_process(self, path):
        tree = ET.ElementTree(ET.fromstring(path))

        for t in tree.iter():
            if "process" in t.tag:
                for child in t:
                    self.__add_node(child)

    def __build_text(self, start, lvl):
        lvl += 1
        self.output.append({"level": lvl, "text": start.name})
        if start.id in self.relationships:
            for dst in self.relationships[start.id]:
                dst_item = self.__search_item_by_id(dst)
                self.__build_text(dst_item, lvl)

    def __search_item_by_id(self, item_id):
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def __add_node(self, child):
        if "id" in child.attrib:
            child_id = child.attrib["id"]
        else:
            child_id = "-1"

        if "name" in child.attrib:
            child_name = child.attrib["name"]
        else:
            child_name = "no name"

        if "startEvent" in child.tag:
            self.items.append(StartEvent(child_id, child_name))
        elif "task" in child.tag:
            self.items.append(Task(child_id, child_name))
        elif "gateway" in child.tag:
            self.items.append(Gateway(child_id, child_name))
        elif "sequenceFlow" in child.tag:
            src = child.attrib["sourceRef"]
            dst = child.attrib["targetRef"]
            if src in self.relationships:
                self.relationships[src].append(dst)
            else:
                self.relationships[src] = [dst]
            self.items.append(SequenceFlow(child_id, child_name))
        else:
            self.items.append(Node(child.tag, child_id, child_name))

class InternalRecomender:

    def __init__(self):
        self.corpus = pd.read_csv("dataset.zip")["description"]
        self.nlp = spacy.load("en_core_web_md")

    def predict(self, bpmn):

        """### Embedding representation

        We will use a pre-trained model to find vector representations of the natural language with acceptable properties.
        """

        parser = Parser(bpmn)

        text = "\n".join(x["text"] for x in parser.output[:-2])

        # Use only a subsample for performance reasons
        SAMPLE_SIZE = 1000

        out = self.corpus.iloc[:SAMPLE_SIZE].apply(self.nlp)

        """The resulting encoding could be stored to enable its rapid use at a later date."""

        # Size is unsuitable for the repo
        # import pickle

        # with open('self.corpus.pickle', 'wb') as f:
        #    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

        """## Search example

        Let's look for the services most similar to the information we have from the diagram (excluding the last stage).
        """

        target = self.nlp(text)

        sims = out.apply(lambda x: target.similarity(x))

        """Note the similarity values are below 0.8. This is because the BPNM text and the services belong to different domains."""
        most_sim = sims.sort_values(ascending=False).iloc[:10].index

        suggestions = pd.DataFrame(self.corpus[most_sim])
        suggestions["sims"] = sims[most_sim]

        """## Domain-related example

        To check the performance of the system we can use one of the dataset services as a search text. We take, for example, the following airline-related service
        """

        target = self.nlp(self.corpus.iloc[96])

        """The suggestions received are related to the search string"""

        sims = out.apply(lambda x: target.similarity(x))
        most_sim = sims.sort_values(ascending=False).iloc[:10].index
        suggestions = pd.DataFrame(self.corpus[most_sim])
        suggestions["sims"] = sims[most_sim]

        suggestions = suggestions.reset_index().rename(columns={'index': 'id'}).to_dict(orient='records')

        return suggestions



global_model = InternalRecomender()

class BPMNItemRecommender:

    def predict(self, bpmn: str) -> List[str]:

        # predict
        global global_model
        recommended_services = global_model.predict(bpmn)

        # format results
        recommended_services.sort(key=lambda x: x['sims'], reverse=True)
        recommended_services = [str(x['id']) for x in recommended_services]

        return recommended_services