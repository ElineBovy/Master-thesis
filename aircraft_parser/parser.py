import stormpy
import inspect
import interval_parser
import numpy as np
from stormpy import *

def export_to_drn(pomdp, export_file):
    stormpy.export_to_drn(pomdp, export_file)


def parse_prism(prism_file, property_string):
    prism_program = stormpy.parse_prism_program(prism_file)

    opts = stormpy.DirectEncodingParserOptions()
    opts.build_choice_labels = True
    properties = stormpy.parse_properties_for_prism_program(property_string, prism_program)
    # construct the pPOMDP
    print(inspect.getfullargspec(stormpy.build_parametric_model))
    pomdp = stormpy.build_parametric_model(prism_program, properties)

    pomdp_parameters = pomdp.collect_probability_parameters()

    return pomdp, pomdp_parameters


def parse_drn(drn_file, property_string):
    drn = stormpy.build_parametric_model_from_drn(drn_file)

    opts = stormpy.DirectEncodingParserOptions()
    opts.build_choice_labels = True
    properties = stormpy.parse_properties_for_prism_program(property_string, drn)
    # construct the pPOMDP
    print(inspect.getfullargspec(stormpy.build_parametric_model))
    pomdp = stormpy.build_parametric_model(drn, properties)

    # get all parameters in the model
    pomdp_parameters = pomdp.collect_probability_parameters()

    return pomdp, pomdp_parameters


def main():
    upomdp_data = []
    # basic idea, load the model with parameters on the transitions where intervals should occur,
    # load a second file mapping these parameters to intervals

    # if you have a drn file instead of a prism file, use parse_drn(...) instead
    #upomdp, params = parse_prism("aircraft_small.prism", "Pmax=?[F \"goal\"]")
    upomdp, params = parse_prism("aircraft_small.prism", "R=?[F \"goal\"]")
    intervals, items = interval_parser.parse_model_interval(upomdp, params, "aircraft_small.intervals")
    
    # loop over the model
    counter = 0
    for state in upomdp.states:
        for action in state.actions:
            lbls = stormpy.ChoiceLabeling.get_labels_of_choice(upomdp.choice_labeling, counter)
            lbl = "deadlock_action"
            if len(lbls) > 0:
                lbl = lbls.pop()
            for transition in action.transitions:
                transition_value = transition.value()
                if transition_value.is_constant():
                    # it's a number!
                    # do whatever you need to do
                    upomdp_data.append([state.id, lbl, transition.column, [int(f"{transition_value}"),int(f"{transition_value}")]])
                    continue
                else:
                    # we assume the denominator is constant, otherwise this breaks
                    # Furthermore, for now, we assume that each transition only depends on at most one interval
                    denom = float(f"{transition_value.denominator}")
                    poly = transition_value.numerator.polynomial()
                    poly_vars = poly.gather_variables()
                    if len(poly_vars) > 1:
                        raise RuntimeError("Too many intervals involved in this transition")
                    num1, num2 = np.zeros(len(poly)), np.zeros(len(poly))
                    for i in range (len(poly)):
                        term = poly[i]
                        if term.is_constant():
                            # if the term is a constant, e.g. 0.5 or 1
                            num1[i], num2[i] = int(f"{term}"), int(f"{term}")
                        else:
                            # we're looking at a variable
                            if not term.monomial[0][0].name == None:
                                name = term.monomial[0][0].name
                                if not str(name) in intervals.keys():
                                    raise RuntimeError("Parameter that was not defined in the interval file")
                                else:
                                    # all looks good, we now have a transition (s,a,s') with the interval given by intervals[term.monomial[0][0].name]
                                    coeff = int(f"{term.coeff}")
                                    
                                    num1[i] = coeff*intervals[name].get_upperbound()
                                    num2[i] = coeff*intervals[name].get_lowerbound()

                            else:
                                raise RuntimeError("Something went wrong with the parameter name")
                                continue
                    b1 = round(num1.sum()/denom,6)
                    b2 = round(num2.sum()/denom,6)
                    lower_bound, upper_bound = (b1, b2) if b1 <= b2 else (b2, b1)
                    upomdp_data.append([state.id, lbl, transition.column, [lower_bound,upper_bound]])
                    interval_str = "[{},{}]".format(lower_bound, upper_bound)
#                     print(f"transition ({state}, {action} (= {lbl}), {transition.column}) with parameter {name} has interval {interval_str}, counter = {counter}")
            counter+=1
    return upomdp, upomdp_data, params

if __name__ == "__main__":
    main()
