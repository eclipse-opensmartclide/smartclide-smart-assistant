from mongoengine import Document
from mongoengine.fields import(
    StringField,
    ListField,
)

class Bpmn(Document):
    name = StringField(Required = True, Unique = True)
    text = StringField(Required=True)
    gherkins = ListField(StringField(Required=True))