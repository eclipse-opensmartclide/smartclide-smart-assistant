import xml.etree.ElementTree as ET


class Node():

    def __init__(self, n_type, n_id, n_name) -> None:
        self.type = n_type
        self.id = n_id
        self.name = n_name

    def __str__(self):
        msg = "------------------ NODO ------------------\n"
        msg += f'ID={self.id}\n'
        msg += f'TYPE={self.type}\n'
        msg += f'NAME={self.name}\n'
        msg += "------------------------------------------\n"
        return msg

class StartEvent(Node):

    def __init__(self, n_id, n_name) -> None:
        super().__init__('startEvent', n_id, n_name)

class Task(Node):

    def __init__(self, n_id, n_name) -> None:
        super().__init__('task', n_id, n_name)

class Gateway(Node):

    def __init__(self, n_id, n_name) -> None:
        super().__init__('gateway', n_id, n_name)

class SequenceFlow(Node):

    def __init__(self, n_id, n_name) -> None:
        super().__init__('sequenceFlow', n_id, n_name)


class Process():
    def __init__(self) -> None:
        self.items = []
        self.relationships = {}
        self.process = ""

    def __str__(self) -> str:
        msg = "\n------------------ PROCESS RELS ------------------\n"
        for rel in self.relationships:
            msg += f'{rel} -> ['
            for dst in self.relationships[rel]:
                msg += f'{dst}, '
            msg += ']\n'
        msg += "---------------------------------------------------\n"
        return msg

    def generate_text_from_bpmn(self, text):
        self.__search_process(text)
        for item in self.items:
            if item.type == 'startEvent':
                self.__build_text(item,0)
                break
        return self.process
    
    def __search_process(self, text):
        tree = ET.ElementTree(ET.fromstring(text))

        for t in tree.iter():
            if 'process' in t.tag:
                for child in t:
                    self.__add_node(child)
    
    #TODO: generate text in the correct format for later NLP
    def __build_text(self, start, lvl):
        lvl += 1
        self.process += f'LVL {lvl} ' + start.name.replace('\n',' ') + ' '
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
        if 'id' in child.attrib:
            child_id = child.attrib['id']
        else:
            child_id = '-1'

        if 'name' in child.attrib:
            child_name = child.attrib['name']
        else:
            child_name = 'no name'

        if 'startEvent' in child.tag:
            self.items.append(StartEvent(child_id, child_name))
        elif 'task' in child.tag:
            self.items.append(Task(child_id, child_name))
        elif 'gateway' in child.tag:
            self.items.append(Gateway(child_id, child_name))
        elif 'sequenceFlow' in child.tag:
            src = child.attrib['sourceRef']
            dst = child.attrib['targetRef']
            if src in self.relationships:
                self.relationships[src].append(dst)
            else:
                self.relationships[src] = [dst]
            self.items.append(SequenceFlow(child_id, child_name))
        else:
            self.items.append(Node(child.tag, child_id, child_name))

    