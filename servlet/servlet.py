from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

server_address = ('0.0.0.0', 8081)
httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
os.chdir('/folder')  # optional
print('Running server...')
httpd.serve_forever()