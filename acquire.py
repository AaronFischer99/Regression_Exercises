
import os
"""Function will extract tables from SQL repository and 
return the telco database which joins tables "internet_service_type",
"contract_types", and "paymeny_types" from telco database
"""
def get_telco_data():
    filename = "telco_data.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        url = env4.get_db_url('telco_churn')

        df_telco = pd.read_sql("select * from customers \
        join internet_service_types using(internet_service_type_id) \
        left join contract_types using(contract_type_id) \
        left join payment_types using(payment_type_id)", env4.get_db_url('telco_churn'))

        df_telco.to_csv(filename)

        return df_telco 

"""Function will extract tables from SQL repository and 
return the titanic database which returns everything from
the "passengers" table
"""   
    
    
import os

def get_titanic_data():
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        url = get_db_url('titanic_db')

        df = pd.read_sql(('SELECT * FROM passengers'), url)

        df.to_csv(filename)

        return df  

    
"""Function will extract tables from SQL repository and 
return the iris database which returns everything from
the "species" and "measurements" table by joining them on 'species_id
"""       
   
import os

def get_iris_data():
    filename = "iris_df.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        url = env4.get_db_url('iris_db')

        df_iris = pd.read_sql("select * from species join measurements using(species_id)", env4.get_db_url('iris_db'))

        df_iris.to_csv(filename)

        return df_iris  

