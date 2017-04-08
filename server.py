from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer
import urlparse, json
from controller import Controller
from Backprop import Backprop
from naive_bayes import naive_bayes


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
        print (data)
        if(len(data['throws']['player1']) >5):
            q = self.model.predict(data)
        self.wfile.write(json.dumps({"nextThrow":"rock","rock":.7,"paper":.2,"scissors":.1}))
        

def run(server_class=HTTPServer, port=3000):
    model = Controller(naive_bayes())
    model.train(.1)
    server_address = ('', port)
    handler = S
    handler.model = model
    httpd = server_class(server_address, handler)
    print 'Starting httpd...'
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()