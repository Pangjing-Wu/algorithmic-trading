import pandas as pd
import psycopg2
from sshtunnel import SSHTunnelForwarder

PORT=5432


remote = dict(
    dbname = r'D:\data\raw\al2\201406\20140603',
    user = 'cra001',
    password = 'cra001',
    host = '10.196.83.198',
    port = '5435',
)



with SSHTunnelForwarder(
    (remote['host'], 22),
    ssh_username='admin',
    ssh_password='FinTech@2020',
    remote_bind_address=('localhost', 5435),
    ) as server:

    conn = psycopg2.connect(
        dbname=r'D:\data\raw\al2\201406\20140603',
        user='cra001',
        password='cra001',
        host='127.0.0.1',
        port=server.local_bind_port
        )

    print(conn)