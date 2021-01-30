import torch
from torch.autograd import Variable

"""sdafasdf"""

import MySQLdb

connection = MySQLdb.connect("localhost", "pi", "abc123", "temps")

with connection.cursor() as cursor:
    # Read from database.
    cursor.execute("SELECT tdate, ttime, zone, temperature FROM tempdat")
    # OR
    # cursor.execute("SELECT * FROM tempdat")

    print("{:15} {:15} {:15} {:15}".
        format("Date", "Time", "Zone", "Temperature"))
    print("===========================================================")

    for reading in cursor.fetchall():
        print("{:15} {:15} {:15} {:15}".
            format(str(reading[0]), str(reading[1]), str(reading[2]), str(reading[3])))

connection.close()

# Change to appropriate username, password and database.
connection = MySQLdb.connect("localhost", "pi", "abc123", "temps")

with connection.cursor() as cursor:
    # Note use of triple quotes for formatting purposes.
    # You can use one set of double quotes if you put the whole string on one line.
    cursor.execute("""INSERT INTO tempdat
        VALUES(CURRENT_DATE() - INTERVAL 1 DAY, NOW(), 'kitchen', 21.7)""")
    cursor.execute("""INSERT INTO tempdat
        VALUES(CURRENT_DATE() - INTERVAL 1 DAY, NOW(), 'greenhouse', 24.5)""")
    cursor.execute("""INSERT INTO tempdat
        VALUES(CURRENT_DATE() - INTERVAL 1 DAY, NOW(), 'garage', 18.1)""")

    cursor.execute("INSERT INTO tempdat VALUES(CURRENT_DATE(), NOW() - INTERVAL 12 HOUR, 'kitchen', 20.6)")
    cursor.execute("INSERT INTO tempdat VALUES(CURRENT_DATE(), NOW() - INTERVAL 12 HOUR, 'greenhouse', 17.1)")
    cursor.execute("INSERT INTO tempdat VALUES(CURRENT_DATE(), NOW() - INTERVAL 12 HOUR, 'garage', 16.2)")

    cursor.execute("INSERT INTO tempdat VALUES(CURRENT_DATE(), NOW(), 'kitchen', 22.9)")
    cursor.execute("INSERT INTO tempdat VALUES(CURRENT_DATE(), NOW(), 'greenhouse', 25.7)")
    cursor.execute("INSERT INTO tempdat VALUES(CURRENT_DATE(), NOW(), 'garage', 18.2)")

connection.commit()
connection.close()


dtype = torch.FloatTensor
N, D_in, H, D_out = 64, 1000, 100, 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # loss = (y_pred - y).pow(2).sum()
    loss = (y_pred - y).sum()
    # print(t, loss.data[0])

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()
