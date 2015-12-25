#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import itertools
    for prediction, age, net_worth in itertools.izip(predictions, ages, net_worths):
        cleaned_data.append((age[0], net_worth[0], abs(prediction[0] - net_worth[0])))
    cleaned_data.sort(key=lambda tup: tup[2])
    nr_to_remove = int((10./100.)*len(cleaned_data))
    for i in range(0, nr_to_remove):
        cleaned_data.pop()
    return cleaned_data

