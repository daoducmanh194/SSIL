def try_index(scalar_of_list, i):
    try:
        return scalar_of_list[i]
    except TypeError:
        return scalar_of_list
        