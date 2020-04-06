# Quick 'n' dirty Python script to listen on a port and do nothing with the connection, simulating a server that allows you to connect but does not reply.
# https://gist.github.com/echojc/5632656
# Originally made for python 2

import socket
import sys

if (len(sys.argv) != 2 or not sys.argv[1].isdigit()):
  print('Usage: listen <port>'),
  exit()

p = int(sys.argv[1])
l = []
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', p))
s.listen(1)
while 1:
  (c, a) = s.accept()
  l.append(c)
  print('%d: connection from %s' % (len(l), a))