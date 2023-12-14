username = 'ursula_2324'

pw = 'i8AUSQ2T792IauwASL5LyIEna2xTKKY1'

host = 'data.codeup.com'

host2 = '123.123.123.123'

url = 'mysql+pymysql://ursula_2324:i8AUSQ2T792IauwASL5LyIEna2xTKKY1@data.codeup.com/employees'

url2 = 'mysql+pymysql://ursula_2324:i8AUSQ2T792IauwASL5LyIEna2xTKKY1@123.123.123.123/employess'



def get_db_url(database,username=username, host=host, password=pw):
    return f'mysql+pymysql://{username}:{password}@{host}/{database}'

