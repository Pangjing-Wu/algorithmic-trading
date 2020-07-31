import sys
sys.path.append('./')

from src.datasource.source import H2Connection, load

remote_args = dict(
    dbdir = r'D:\data\raw\al2\201406',
    user = 'cra001',
    password = 'cra001',
    host = '10.196.83.198'
)

def test_remote_connection():
    h2 = H2Connection(**remote_args)
    assert h2.status != False

test_remote_connection()