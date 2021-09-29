"""
Module providing the functionality to define case bases
"""

import pandas as pd
import os
from mongoengine import connect , errors
from  pycbr.documents import Bpmn
from pycbr.builder import Process


class CaseBase:
    def __init__(self):
        pass

    def get_pandas(self):
        """
        Get a pandas dataframe representing the case base

        Returns:
            pandas.DataFrame: A dataframe representing the case base.

        """
        raise NotImplementedError

    def add_case(self, case, case_id=None):
        """
        Add a case to the case base

        Args:
            case: A description of the case.
            case_id: An identifier of the case.

        """
        raise NotImplementedError

    def delete_case(self, case_id):
        """
        Remove a case from the case base

        Args:
            case_id: Unique identifier of the case

        """
        raise NotImplementedError

    def get_description(self):
        """Get a dictionary describing the instance"""
        raise NotImplementedError


class MongoCaseBase(CaseBase):
    """A case base defined from a pandas dataframe to mongodb"""

    def __init__(self, cases, db , host):
        """
        Args:
            cases: List of cases each element needs text, file, gherkins
        """
        super().__init__()
        connect(db=db, host="localhost")
        for row in cases:
            try:
                Bpmn(name=row['name'],text=row['text'],gherkins=row['gherkins']).save()
            except errors.NotUniqueError as e:
                pass
        self.header = ['name','text','gherkins']

    def get_description(self):
        return {"__class__": self.__class__.__module__ + "." + self.__class__.__name__,
                "cases": "<<List>>"}

    def get_pandas(self):
        """ Get all cases and create a df"""
        df = pd.DataFrame([],columns=self.header)
        for case in Bpmn.objects():
            df = df.append({"name": case.name,"text": case.text,"gherkins": case.gherkins}, ignore_index= True)
        return df

    def add_case(self, case, case_id=None):
        """ Add case to database """
        process = Process()
        Bpmn(name = case['name'],text = process.generate_text_from_bpmn(case['text']),gherkins = case['gherkins']).save()

    def delete_case(self, case_id):
        """ Delete case from database """
        case = Bpmn.objects()[case_id]
        os.remove(case.name)
        case.delete()
        
    def casemaker(self, file):
        process = Process()
        return {'name': 'None' , 'text': process.generate_text_from_bpmn(file) , 'gherkins': ['none'] }
    
    