"""pycbr - Package to implement Case-Based Reasoning systems"""

__version__ = '0.1.1'
__author__ = 'Dih5 <dihedralfive@gmail.com>'
import pandas as pd

from . import aggregate, builder, models, recovery, casebase, documents


class CBR:
    """A CBR application"""

    def __init__(self, cases, db , host, refit_always=True):
        """

        Args:
            case: list of cases
            db: database
            host: database host
        """
        self.case_base = casebase.MongoCaseBase(cases,db,host)
        self.recovery_model = recovery.Recovery([("text", models.TextAttribute())],algorithm="brute")
        self.aggregator = aggregate.MajorityAggregate("gherkins")
        self.refit_always = refit_always

        self._int_index = None

        self.refit()


    def refit(self):
        """Update the recovery model to match the case base"""
        df = self.case_base.get_pandas()
        self._int_index = df.index.dtype in ["int16", "int32", "int64"]
        self.recovery_model.fit(df)

    def get_pandas(self):
        """
        Get a pandas dataframe representing the case base

        Returns:
            pandas.DataFrame: A dataframe representing the case base.

        """
        return self.case_base.get_pandas()

    def get_case(self, case_id):
        """
        Retrieve a case with the given id

        Args:
            case_id: An identifier of the id.

        Returns:
            pandas.Series: The case found.

        """
        df = self.get_pandas()
        if self._int_index:
            case_id = int(case_id)
        return df.loc[case_id]

    def add_case(self, case, case_id=None):
        """
        Add a case to the case base

        Args:
            case: A description of the case.
            case_id: An identifier of the case.
        """
        if case_id is not None and self._int_index:
            case_id = int(case_id)
        r = self.case_base.add_case(case, case_id=case_id)
        if self.refit_always:
            self.refit()
        return r

    def delete_case(self, case_id):
        """
        Remove a case from the case base

        Args:
            case_id: Unique identifier of the case

        """
        if self._int_index:
            case_id = int(case_id)
        r = self.case_base.delete_case(case_id)
        if self.refit_always:
            self.refit()
        return r
    
    def recommend(self, file):
        case = self.case_base.casemaker(file)
        df_sim, sims = self.recovery_model.find(pd.DataFrame([case]), 5)[0]
        cases = [row.dropna().to_dict() for _, row in df_sim.iterrows()]
        gherkins = [case['gherkins'] for case in cases]
        return {"gherkins": gherkins,
                "sims": sims.tolist()}
