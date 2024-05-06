import prettytable as pt

def pretty_table_log(field, values):
    tb = pt.PrettyTable()
    tb.field_names = field
    tb.add_row(values)
    print(tb)