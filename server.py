from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import urlparse, json
from controller import Controller
from Backprop import Backprop

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
    def do_POST(self):
        # Doesn't do anything with posted data
        content_len = int(self.headers.getheader('content-length', 0))
        post_body = self.rfile.read(content_len)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        
        data = json.loads(post_body)
        
        self.wfile.write(json.dumps({"rock":.7,"paper":.2,"scissors":.1}))
        
        
def run(server_class=HTTPServer, handler_class=S, port=8080):
    q = Controller(Backprop())
    q.train()
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()