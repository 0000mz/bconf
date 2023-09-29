from bconf_lib import GrammarTree, Token

def serve_parse_tree(tree: GrammarTree, tokenlst: list[Token]):
    import logging
    import matplotlib.pyplot as plt
    import networkx as nx # type: ignore
    from io import BytesIO
    import base64

    def get_token_string(tokenlst: list[Token], index: int) -> str:
        if index < 0 or index >= len(tokenlst):
            return "None"
        return str(tokenlst[index])

    plt.rcParams["figure.figsize"] = [10, 10]
    G = nx.Graph()

    label_dict = {}
    color_dict = {}

    stack = [tree.root]
    while stack:
        next_node = stack.pop()
        label_dict[next_node.id] = str(next_node) + ", token = " + get_token_string(tokenlst, next_node.token_index)
        color_dict[next_node.id] = "blue" if next_node.valid else "red"

        for child_node in next_node.children:
            stack.append(child_node)
            G.add_edge(next_node.id, child_node.id)

    color_seq = [color_dict[node] for node in G]
    nx.draw(G, labels=label_dict, node_color=color_seq, with_labels=True)
    plt.title("Parse Tree")

    buf = BytesIO()
    plt.savefig(buf, format="png")

    data = base64.b64encode(buf.getbuffer())    
    html = bytes("<img src='data:image/png;base64,", 'utf-8') + data + bytes("'/>", 'utf-8')

    from http.server import HTTPServer, BaseHTTPRequestHandler
    class StaticServer(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html)

    def run_server(server_class=HTTPServer, handler_class=StaticServer, port=8000):
        server_address = ('', port)
        httpd = server_class(server_address, handler_class)
        logging.info("Serving parse tree to http://localhost:%s", port)
        httpd.serve_forever()

    run_server()

