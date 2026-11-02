"""pandas 3.1 select vs assign: schema, expression order and filtering."""
import json
from pathlib import Path
import sys
import pandas as pd

print('pandas', pd.__version__)
orders = pd.DataFrame({'order': ['A', 'B', 'C'], 'qty': [2, 3, 1],
                       'price': [10., 20., 30.], 'internal_note': ['x', 'y', 'z']},
                      index=[10, 20, 40])
selected = orders.select('order', gross=pd.col('qty') * pd.col('price'),
                         tax=pd.col('gross') * 0.1)
assigned = orders.assign(gross=pd.col('qty') * pd.col('price'),
                         tax=pd.col('gross') * 0.1)
pd.testing.assert_frame_equal(selected, assigned[['order', 'gross', 'tax']])
print('select columns:', selected.columns.tolist())
print('assign columns:', assigned.columns.tolist())
print(selected.to_string())
print('source unchanged:', orders.columns.tolist())

deferred = orders.assign(qty=pd.col('qty') + 1,
                          gross=pd.col('qty') * pd.col('price'))
eager = orders.assign(qty=orders['qty'] + 1,
                       gross=orders['qty'] * orders['price'])
print('new qty:', deferred['qty'].tolist())
print('deferred gross:', deferred['gross'].tolist())
print('eager gross:', eager['gross'].tolist())
assert deferred['gross'].tolist() == [30., 80., 60.]
assert eager['gross'].tolist() == [20., 60., 30.]
filtered = (orders.filter(pd.col('qty') > 1)
            .select('order', gross=pd.col('qty') * pd.col('price')))
pd.testing.assert_frame_equal(filtered, orders.loc[orders['qty'] > 1]
                              .assign(gross=lambda d: d.qty * d.price)[['order', 'gross']])
print('filtered indices:', filtered.index.tolist())
print('PASS: equivalent projection; original preserved; expression dependency checked')
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(dict(
        source=orders.columns.tolist(), select=selected.columns.tolist(),
        assign=assigned.columns.tolist(), rows=selected.to_dict('records'),
        deferred=deferred['gross'].tolist(), eager=eager['gross'].tolist()), indent=2) + '\n')
