from tabulate import tabulate

def create_small_table(data):
    keys, values = tuple(zip(*data.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table

def create_multi_column_table(data, num_column, headers = [], align='center'):
    table = tabulate(
            data,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=headers * num_column,
            numalign=align,
        )

    return table