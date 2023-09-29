import sys
import server_util
import logging
import argparse
from bconf_lib import Parser, InputFileStream

if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument('file', help="bconf file to parse.")
    aparser.add_argument('--debug', help="Enable debug mode.", action='store_true')
    aparser.add_argument(
        '--serve_parse_tree',
        help="After parsing is complete, serve the final parse tree as a pyvis interface for debugging.",
        action='store_true'
    )

    args = aparser.parse_args()
    logging.getLogger(None).setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.debug("parsing input file: %s", args.file)

    bconf_parser = Parser(InputFileStream.FromFilename(args.file))
    parse_success = bconf_parser.parse()

    if args.serve_parse_tree:
        assert bconf_parser.tree is not None
        server_util.serve_parse_tree(bconf_parser.tree, bconf_parser.tokens)

    if not parse_success:
        logging.error("Parser failed.")
        sys.exit(1)

