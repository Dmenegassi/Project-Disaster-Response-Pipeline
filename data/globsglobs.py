from sqlalchemy import create_engine

import pandas as pd
conn = connect(':memory:')
db = pd.read_sql('SELECT * FROM YourDatabaseName', conn)
print(db)
                 
                 