import time
from l2l.utils.groups import ParameterGroup, ResultGroup, ParameterDict
from l2l.utils.individual import Individual
import logging

logging = logging.getLogger("Trajectory")


class Trajectory:
    """
    The trajectory is a class which holds the history of the parameter space exploration, defines the current
    parameters to be explored and holds the results from each execution.
    Based on the pypet trajectory concept: https://github.com/SmokinCaterpillar/pypet
    """

    def __init__(self, **keyword_args):
        """
        Initializes the trajectory. Some parameters are kept to match the interface with the pypet trajectory.
        """
        if 'name' in keyword_args:
            self._name = keyword_args['name']
        self._timestamp = time.time()
        self.parameters = ParameterDict(self)  # Contains all parameters
        self.results = ResultGroup()
        self.current_results = {}
        self.individual = Individual()
        self.individuals = {}
        self.v_idx = 0

    def f_add_parameter_group(self, name, comment=""):
        """
        Adds a new parameter group
        :param name: name of the new parameter group
        :param comment: ignored for the moment. Kept to match pypet interface.
        """
        self.parameters[name] = ParameterGroup()
        logging.info("Added new parameter group: " + name)

    def f_add_parameter_to_group(self, group_name, key, val):
        """
        Adds a parameter to an already existing group.

        :param group_name: Name of the group where the parameter should be added
        :param key: Name of the parameter to be added
        :param val: Value of the parameter

        Throws an exception if the group does not exist
        """
        if group_name in self.parameters.keys():
            self.parameters[group_name].f_add_parameter(key, val)
        else:
            # LOG("Key not found when adding to result group")
            raise Exception("Group name not found when adding value to result group")

    def f_add_result(self,key, val, comment=""):
        """
        Adds a result to the trajectory
        :param key: it identifies either a generation params result group or another result
        :param val: The value to be added to the results
        """
        self.results[key] = val

    def f_add_parameter(self, key, val, comment=""):
        """
        Adds a parameter to the trajectory
        :param key: Name of the parameter
        :param val: Value of the parameter
        :param comment
        """
        self.parameters[key] = val

    def f_expand(self, build_dict, fail_safe=True):
        """
        The expand function takes care of adding a new generation and individuals to the trajectory
        This is a critical function to allow the addition of a new generation, called by the optimizer
        from the postprocessing function
        :param build_dict: The dictionary containing the new generation id and its individuals
        :param fail_safe: Currently ignored
        """
        params = {}
        gen = []
        ind_idx = []
        for key in build_dict.keys():
            if key == 'generation':
                gen = build_dict['generation']
            elif key == 'ind_idx':
                ind_idx = build_dict['ind_idx']
            else:
                params[key] = build_dict[key]

        generation = gen[0]
        self.individuals[generation] = []

        for i in ind_idx:
            ind = Individual(generation,i,[])
            for j in params:
                ind.f_add_parameter(j, params[j][i])
            self.individuals[generation].append(ind)
        logging.info("Expanded trajectory for generation: " + str(generation))

    def __str__(self):
        return str(self.parameters)

    def __getattr__(self, attr):
        """
        Handle attribute access like a sdict
        :param attr: The attribute to be accessed
        :return: the value of this attributes
        """
        if '.' in attr:
            # This is triggered exclusively in the case where __getattr__ is called from __getitem__
            attrs = attr.split('.')
            ret = self.parameters.get(attrs[0])
            for at in attrs[1:]:
                ret = ret[at]
        elif attr == 'parameters':
            ret = self.parameters
        else:
            ret = self.parameters.get(attr,default_value=None)
        return ret

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
